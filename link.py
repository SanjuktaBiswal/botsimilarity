import streamlit as st
import pandas as pd
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, CustomJS
from bokeh.models import DataTable, TableColumn, HTMLTemplateFormatter
from streamlit_bokeh_events import streamlit_bokeh_events


df = pd.DataFrame({
        "links": ["https://www.google.com", "https://streamlit.io", "https://outlokk"],
        "vendor": ["a", "b", "c"],
        "pickup": ["a1", "b1", "c1"],
})

df.style.set_properties(**{'background-color': 'black',
       'color': 'green'})
st.dataframe(df.style.set_properties(**{'background-color': 'black',
       'color': 'green'}))
# create plot
cds = ColumnDataSource(df)

columns = [
TableColumn(field="links", title="ID", formatter=HTMLTemplateFormatter(template='<a href="";target="_blank"><%= value %>')),
TableColumn(field="vendor", title="Vendor"),
TableColumn(field="pickup", title="Pickup", formatter=HTMLTemplateFormatter(template='<a href="";target="_blank"><%= value %>')),
]

# define events
cds.selected.js_on_change(
"indices",
CustomJS(
        args=dict(source=cds),
        code="""
        document.dispatchEvent(
        new CustomEvent("INDEX_SELECT", {detail: {data: source.selected.indices}})
        )
        """
)
)
p = DataTable(source=cds, columns=columns, css_classes=["my_table"])

result= streamlit_bokeh_events(bokeh_plot=p, events="INDEX_SELECT", key="foo", refresh_on_update=False, debounce_time=0, override_height=100)
if result!=None:
    st.text("Click on Row No:")
    st.text(result[25:26])
