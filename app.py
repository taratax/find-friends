import streamlit as st
import json
import pandas as pd
from pycaret.clustering import load_model, predict_model
import plotly.express as px
import plotly.graph_objects as go
import random

MODEL_NAME = 'welcome_survey_clustering_pipeline_v245'
DATA = 'welcome_survey_simple_v2.csv'
CLUSTER_NAMES_AND_DESCRIPTIONS = 'welcome_survey_cluster_names_and_descriptions_v245.json'

@st.cache_resource
def get_model():
    return load_model(MODEL_NAME)

@st.cache_data
def get_cluster_names_and_descriptions():
    with open(CLUSTER_NAMES_AND_DESCRIPTIONS, "r") as f:
        return json.loads(f.read())

@st.cache_data
def get_all_participants():
    model = get_model()
    all_df = pd.read_csv(DATA, sep=';')
    df_with_clusters = predict_model(model, data=all_df)
    return df_with_clusters

st.title("Znajdź znajomych")

with st.sidebar:
    st.header("Powiedz nam coś o sobie")
    st.markdown("Pomożemy Ci znaleźć osoby, które mają podobne zainteresowania")

    age = st.selectbox("Wiek", ['<18', '25-34', '45-54', '35-44', '18-24', '>=65'])
    edu_level = st.selectbox("Wykształcenie", ['Podstawowe', 'Średnie', 'Wyższe'])
    fav_animals = st.selectbox("Ulubione zwierzęta", ["Brak ulubionych", "Psy", "Koty", "Psy i koty"])
    fav_place = st.selectbox("Ulubione miejsce", ["Nad wodą", "W lesie", "W górach", "Inne"])
    gender = st.radio("Płeć", ["Mężczyzna", "Kobieta"])

    person_df = pd.DataFrame([{
        'age': age,
        'edu_level': edu_level,
        'fav_animals': fav_animals,
        'fav_place': fav_place,
        'gender': gender
    }])

model = get_model()
all_df = get_all_participants()
cluster_names_and_descriptions = get_cluster_names_and_descriptions()

predicted_cluster_id = predict_model(model, data=person_df)["Cluster"].values[0]
predicted_cluster_data = cluster_names_and_descriptions[str(predicted_cluster_id)]

st.header(f"Najbliżej Ci do grupy: {predicted_cluster_data['name']}")
st.markdown(predicted_cluster_data['description'])

same_cluster_df = all_df[all_df["Cluster"] == predicted_cluster_id]
st.metric("Liczba Twoich znajomych", len(same_cluster_df))

st.header("Osoby z grupy")

fig = px.histogram(same_cluster_df.sort_values("age"), x="age")
fig.update_layout(
    title="Rozkład wieku w grupie",
    xaxis_title="Wiek",
    yaxis_title="Liczba osób",
)
st.plotly_chart(fig)

fig = px.histogram(same_cluster_df, x="edu_level")
fig.update_layout(
    title="Rozkład wykształcenia w grupie",
    xaxis_title="Wykształcenie",
    yaxis_title="Liczba osób",
)
st.plotly_chart(fig)

fig = px.histogram(same_cluster_df, x="fav_animals")
fig.update_layout(
    title="Rozkład ulubionych zwierząt w grupie",
    xaxis_title="Ulubione zwierzęta",
    yaxis_title="Liczba osób",
)
st.plotly_chart(fig)

fig = px.histogram(same_cluster_df, x="fav_place")
fig.update_layout(
    title="Rozkład ulubionych miejsc w grupie",
    xaxis_title="Ulubione miejsce",
    yaxis_title="Liczba osób",
)
st.plotly_chart(fig)

st.header("Porównanie wspólnych cech uczestników")

edu_level_counts = same_cluster_df['edu_level'].value_counts(normalize=True)
age_counts = same_cluster_df['age'].value_counts(normalize=True)


categories = ['Podstawowe', 'Średnie', 'Wyższe', '<18', '18-24', '25-34', '35-44', '45-54', '>=65']
values = [
    edu_level_counts.get('Podstawowe', 0),
    edu_level_counts.get('Średnie', 0),
    edu_level_counts.get('Wyższe', 0),
    age_counts.get('<18', 0),
    age_counts.get('18-24', 0),
    age_counts.get('25-34', 0),
    age_counts.get('35-44', 0),
    age_counts.get('45-54', 0),
    age_counts.get('>=65', 0)
]

fig = go.Figure()
fig.add_trace(go.Scatterpolar(
    r=values,
    theta=categories,
    fill='toself'
))
fig.update_layout(
    polar=dict(
        radialaxis=dict(visible=True)
    ),
    title="Radar cech wspólnych w grupie"
)
st.plotly_chart(fig)

st.header("Ciekawostka o Twojej grupie")

most_common_place = same_cluster_df['fav_place'].mode()[0]
most_common_animal = same_cluster_df['fav_animals'].mode()[0]
most_common_age_group = same_cluster_df['age'].mode()[0]

fun_facts = [
    f"Czy wiesz, że większość osób w tej grupie preferuje spędzać czas w miejscu: {most_common_place}?",
    f"Ulubionym zwierzęciem większości osób w tej grupie są: {most_common_animal}.",
    f"Większość osób w tej grupie jest w wieku: {most_common_age_group}."
]

st.markdown(random.choice(fun_facts))

st.header("Rekomendacja aktywności grupowych")

if most_common_place == "Nad wodą":
    recommended_activity = "Wypad na kajaki lub żeglowanie po Mazurach."
elif most_common_place == "W lesie":
    recommended_activity = "Spacer lub wycieczka rowerowa po Puszczy Białowieskiej."
elif most_common_place == "W górach":
    recommended_activity = "Wyprawa górska w Tatry."
else:
    recommended_activity = "Zwiedzanie i atrakcje miejskie w Warszawie."

st.subheader("Proponowana aktywność dla Twojej grupy:")
st.markdown(recommended_activity)
