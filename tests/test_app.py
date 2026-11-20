from pathlib import Path
import pytest

def test_cached_demo_navigation_and_profile_edits(monkeypatch):
    root=Path(__file__).resolve().parents[1]
    artifacts=root/'results/smoke'
    if not (artifacts/'demo_bundle.pkl').exists():
        pytest.skip('Run the smoke experiment to enable cached-demo integration checks')
    monkeypatch.setenv('TASTETRACE_RESULTS',str(artifacts))
    from streamlit.testing.v1 import AppTest
    app=AppTest.from_file(str(root/'app/streamlit_app.py'),default_timeout=60).run()
    assert not app.exception
    for page in ['Side-by-side comparison','Results dashboard','How it works','Live recommender']:
        app.selectbox(key='page').select(page).run()
        assert not app.exception
    app.segmented_control(key='profile_mode').set_value('Build a profile').run()
    assert not app.exception
    app.button(key='add_to_profile').click().run()
    assert not app.exception
    assert app.session_state.history==[1]
    app.button(key='remove_from_profile').click().run()
    assert app.session_state.history==[]
