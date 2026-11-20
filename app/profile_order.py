"""Offline CCv2 draggable list with keyboard reorder controls."""
import streamlit as st

_ORDER=st.components.v2.component(
    'tastetrace_profile_order',
    html="""
    <p>Drag movies into viewing order, or use the arrows. Earliest comes first.</p>
    <ol id="movies" aria-label="Viewing order"></ol>
    <button id="apply" type="button">Apply dragged order</button>
    """,
    js="""
export default function(component) {
  const {parentElement, data, setTriggerValue} = component;
  const list = parentElement.querySelector('#movies');
  list.replaceChildren();
  let dragged = null;
  for (const movie of data.movies) {
    const row = document.createElement('li');
    row.draggable = true; row.dataset.id = movie.id;
    const label = document.createElement('span');
    label.textContent = movie.title + ' '; row.appendChild(label);
    for (const [text, direction] of [['↑', -1], ['↓', 1]]) {
      const button = document.createElement('button');
      button.type = 'button'; button.textContent = text;
      button.setAttribute('aria-label', (direction < 0 ? 'Move earlier: ' : 'Move later: ') + movie.title);
      button.onclick = () => {
        if (direction < 0 && row.previousElementSibling) list.insertBefore(row, row.previousElementSibling);
        if (direction > 0 && row.nextElementSibling) list.insertBefore(row.nextElementSibling, row);
      };
      row.appendChild(button);
    }
    row.ondragstart = e => { dragged = row; e.dataTransfer.setData('text/plain', row.dataset.id); };
    row.ondragover = e => e.preventDefault();
    row.ondrop = e => { e.preventDefault(); if (dragged && dragged !== row) list.insertBefore(dragged, row); };
    list.appendChild(row);
  }
  parentElement.querySelector('#apply').onclick = () => {
    setTriggerValue('submitted', Array.from(list.children, row => Number(row.dataset.id)));
  };
}
    """,
)

def profile_order(movies,key):
    result=_ORDER(data={'movies':movies},key=key,on_submitted_change=lambda:None)
    return result.submitted
