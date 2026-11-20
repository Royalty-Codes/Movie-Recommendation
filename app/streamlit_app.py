from pathlib import Path
import os
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'app'))
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from tastetrace.demo_utils import load_bundle,recommend
from tastetrace.explain import explain
from tastetrace.hmm import user_states
from profile_order import profile_order

st.set_page_config(page_title='TasteTrace',page_icon=':material/movie:',layout='wide')
st.title('TasteTrace')
st.caption('Follow your changing movie taste. See the evidence behind every recommendation.')
results=Path(os.environ.get('TASTETRACE_RESULTS',str(ROOT/'results')))
if not (results/'demo_bundle.pkl').exists():
    st.info('Train the models first with `python scripts/run_all.py`. For the integration demo, set TASTETRACE_RESULTS to results/smoke.')
    st.stop()

@st.cache_resource
def cached_bundle(path,modified):
    return load_bundle(path)

bundle=cached_bundle(str(results/'demo_bundle.pkl'),(results/'demo_bundle.pkl').stat().st_mtime_ns)
tt=bundle['models']['TasteTrace-v2']
movies=tt.movies; cfg=bundle['config']
page=st.sidebar.selectbox('Explore',['Live recommender','Side-by-side comparison','Results dashboard','How it works'],key='page')
users=sorted(bundle['split'].test.user_id.tolist())
uid=st.sidebar.selectbox('Viewer',users,index=users.index(42) if 42 in users else 0,key='viewer')
st.sidebar.caption(f'Experiment {bundle["config_hash"][:8]} · {cfg["dataset"]}')
if results.name=='smoke': st.warning('Integration demo on a reduced dataset. These are not final research results.')

if page=='Live recommender':
    st.subheader('Your viewing journey')
    mode=st.segmented_control('Profile',['Existing viewer','Build a profile'],default='Existing viewer',key='profile_mode')
    baseline=list(map(int,tt.index.histories[uid])) if mode=='Existing viewer' else []
    title=lambda i:f'{movies.at[i,"title"]} · #{i}'
    if 'profile_id' not in st.session_state or st.session_state.profile_id!=(uid,mode):
        st.session_state.profile_id=(uid,mode); st.session_state.history=baseline.copy()
    history=st.session_state.history
    with st.container(horizontal=True):
        add=st.selectbox('Add a movie at the end',movies.index.tolist(),format_func=title,key='add_movie')
        if st.button('Add movie',key='add_to_profile') and add not in history:
            history.append(add); st.rerun()
        if st.button('Reset profile',key='reset_profile'):
            st.session_state.history=baseline.copy(); st.rerun()
    if history:
        remove=st.selectbox('Remove a watched movie',history,format_func=title,key='remove_movie')
        if st.button('Remove movie',key='remove_from_profile'):
            history.remove(remove); st.rerun()
        st.caption('Edit the order numbers to reorder a profile; the sequence below runs from earliest to latest.')
        order_frame=pd.DataFrame({'item_id':history,'Movie':[movies.at[i,'title'] for i in history],'Order':list(range(1,len(history)+1))})
        edited=st.data_editor(order_frame,disabled=['item_id','Movie'],hide_index=True,key=f'order_{uid}_{mode}_{len(history)}',alt='Ordered watched movies')
        if st.button('Apply viewing order',key='apply_order'):
            st.session_state.history=edited.sort_values('Order',kind='stable').item_id.astype(int).tolist(); st.rerun()
        if mode=='Build a profile':
            dragged=profile_order([{'id':i,'title':movies.at[i,'title']} for i in history],
                                  key='drag_'+str(uid)+'_'+'_'.join(map(str,history)))
            if dragged is not None and dragged!=history and sorted(dragged)==sorted(history):
                st.session_state.history=dragged; st.rerun()
    custom=history!=baseline or mode=='Build a profile'
    hist_arg=history if custom else None
    paths,post,nxt=user_states(tt.hmm,tt.index,tt.labels,{uid:np.array(history,dtype=int)})
    state=int(post[uid].argmax()); cluster=int(tt.hmm.emissionprob_[state].argmax())
    st.metric('Current inferred phase',tt.names[cluster]['name'])
    st.caption('Phases describe patterns of movie viewing; they are not causal claims about a person.')
    if history:
        timeline=pd.DataFrame({'Watch order':np.arange(1,len(history)+1),'Phase':paths[uid].astype(str),
                               'Movie':[movies.at[i,'title'] for i in history]})
        chart=alt.Chart(timeline).mark_circle(size=70).encode(x='Watch order:Q',y='Phase:N',color='Phase:N',tooltip=['Movie','Watch order','Phase'])
        st.altair_chart(chart,alt='Viewing timeline colored by inferred phase')
    left,right=st.columns(2)
    with left:
        st.bar_chart(pd.DataFrame({'Phase':np.arange(len(nxt[uid])).astype(str),'Probability':nxt[uid]}),x='Phase',y='Probability',alt='Predicted next phase probabilities')
    with right:
        edges=list(tt.networks[state].edges())
        dot='digraph phase { '+ ';'.join(f'"{a}" -> "{b}"' for a,b in edges)+';'+ ';'.join(f'"{n}"' for n in tt.networks[state].nodes())+'; }'
        st.graphviz_chart(dot,alt='Learned Bayesian network for current phase')
    st.subheader('Recommended for this profile')
    for rank,(item,score) in enumerate(recommend(tt,uid,history=hist_arg),1):
        ex=explain(tt,uid,item,hist_arg)
        with st.container(border=True):
            st.markdown(f'**{rank}. {movies.at[item,"title"]}**')
            st.write(ex['text'])
            st.bar_chart(pd.DataFrame({'Component':list(ex['contributions']),'Contribution':list(ex['contributions'].values())}),x='Component',y='Contribution',horizontal=True,alt=f'Score contributions for {movies.at[item,"title"]}')
    if custom: st.caption('What-if profiles use frozen trained models and SVD fold-in; they do not retrain or alter the research results.')

elif page=='Side-by-side comparison':
    st.subheader('One viewer, three recommenders')
    target=int(bundle['split'].test.set_index('user_id').at[uid,'item_id'])
    st.write('Held-out movie: **'+movies.at[target,'title']+'**')
    names=[bundle['stronger_paper'],'SVD-MF','TasteTrace-v2']
    lists={n:recommend(bundle['models'][n],uid) for n in names}
    common=set(i for i,_ in lists[names[0]])&set(i for i,_ in lists[names[1]])&set(i for i,_ in lists[names[2]])
    for col,name in zip(st.columns(3,border=True),names):
        with col:
            st.subheader(name)
            rank=next((j+1 for j,(i,_) in enumerate(lists[name]) if i==target),None)
            st.caption(f'Held-out rank: {rank}' if rank else 'Held-out movie absent from top 10')
            for r,(i,_) in enumerate(lists[name],1):
                st.markdown(f'{r}. {movies.at[i,"title"]}'+(' ★' if i==target else '')+(' · shared' if i in common else ''))
                if name=='TasteTrace-v2': st.caption(explain(tt,uid,i)['text'])

elif page=='Results dashboard':
    st.subheader('Measured results')
    if (results/'SUMMARY.md').exists(): st.markdown((results/'SUMMARY.md').read_text(encoding='utf-8'))
    for filename in ['main_comparison.csv','ablation.csv','faithfulness.csv','significance.csv','beyond_accuracy.csv']:
        path=results/filename
        if path.exists():
            st.subheader(filename.replace('_',' ').replace('.csv',''))
            st.dataframe(pd.read_csv(path),hide_index=True,alt=f'Computed {filename} results')
    figures=ROOT/cfg['figures_dir']
    for name in ['hero_comparison','ablations','faithfulness','protocol_flaw']:
        if (figures/f'{name}.png').exists(): st.image(str(figures/f'{name}.png'),alt=name.replace('_',' '))

else:
    st.subheader('From viewing history to an explanation')
    st.graphviz_chart('digraph pipeline { rankdir=LR; "Train interactions" -> "SVD taste clusters" -> "HMM taste phases" -> "Per-phase Bayesian networks" -> "Six score components" -> "Validation-trained linear fusion" -> "Recommendation + evidence"; }',alt='TasteTrace model pipeline')
    st.write('Collaborative clusters describe movies watched by similar audiences. A shared HMM tracks changes over an ordered history. Each phase has a Bayesian network describing combinations of item attributes. A nonnegative linear ranker combines these signals with recent affinity, item similarity, matrix factorization and popularity.')
    st.write('Explanations cite positive score contributions and real watched titles. Component deletion tests measure whether those cited contributions affect rank. The held-out test movie never trains the feature pipeline or ranker.')
    st.info('Bayesian-network edges describe statistical dependencies. Observational MovieLens data does not establish causal effects.')
