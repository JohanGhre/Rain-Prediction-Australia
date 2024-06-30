import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, classification_report, roc_curve
from datetime import datetime

# Set Up Streamlit App 
st.set_page_config(page_title="Raining Prediction in Australia",
                   page_icon="🌤️",
                   layout="wide")

# Charger le modèle et les transformateurs
aussie_rain = joblib.load('aussie_rain.joblib')

model = aussie_rain['model']
imputer = aussie_rain['imputer']
scaler = aussie_rain['scaler']
encoder = aussie_rain['encoder']
input_cols = aussie_rain['input_cols']
target_col = aussie_rain['target_col']
numeric_cols = aussie_rain['numeric_cols']
categorical_cols = aussie_rain['categorical_cols']
encoded_cols = aussie_rain['encoded_cols']

# Fonction pour prédire la pluie
def predict_input(single_input):
    input_df = pd.DataFrame([single_input])
    input_df[numeric_cols] = imputer.transform(input_df[numeric_cols])
    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])
    input_df[encoded_cols] = encoder.transform(input_df[categorical_cols])
    X_input = input_df[numeric_cols + encoded_cols]
    pred = model.predict(X_input)[0]
    prob = model.predict_proba(X_input)[0][list(model.classes_).index(pred)]
    return pred, prob

# Chargement des données d'entraînement pour la visualisation
train_df = pd.read_parquet('train_inputs.parquet')
train_df[target_col] = pd.read_parquet('train_targets.parquet')[target_col]

# Affichage du titre dans l'interface utilisateur
st.title("Rainfall Prediction Australie 🌧️🇦🇺")

# Mise en page à deux colonnes avec ajustement de la largeur
col1, col2 = st.columns([2, 1])

# Colonne 1 - Visualisation des données et informations
with col1:

    st.header("Data Visualization")
    
    # Histogramme des températures maximales
    fig = px.histogram(train_df, x='MaxTemp', nbins=50, title='Distribution des Températures Maximales',
                       color_discrete_sequence=px.colors.qualitative.Pastel, width=600, height=600)
    st.plotly_chart(fig)
    
    # Boxplot des précipitations par cible
    fig = px.box(train_df, x=target_col, y='Rainfall', title='Précipitations par Prévision de Pluie',
                 color_discrete_sequence=px.colors.qualitative.Pastel)
    st.plotly_chart(fig)
    
    # Scatter plots
    fig = px.scatter(train_df.sample(5000), 
                     title='Temp3pm vs. Humidity3pm',
                     x='Temp3pm', 
                     y='Humidity3pm', 
                     color='RainTomorrow',
                     color_discrete_sequence=px.colors.qualitative.Plotly)
    st.plotly_chart(fig)

    fig = px.scatter(train_df.sample(5000), 
                     title='Min Temp. vs Max Temp.',
                     x='MinTemp', 
                     y='MaxTemp', 
                     color='RainToday',
                     color_discrete_sequence=px.colors.qualitative.Plotly)
    st.plotly_chart(fig)

    # Histogram plots
    fig = px.histogram(train_df, 
                       x="RainTomorrow", 
                       title="RainToday vs. RainTomorrow", 
                       color="RainToday",
                       color_discrete_sequence=px.colors.qualitative.Plotly)
    st.plotly_chart(fig)

    fig = px.histogram(train_df, 
                       x="Temp3pm", 
                       title="Temp3pm vs. RainTomorrow", 
                       color="RainTomorrow", 
                       color_discrete_sequence=px.colors.qualitative.Plotly)
    st.plotly_chart(fig)

    fig = px.histogram(train_df, 
                       x="Location", 
                       title="Location vs. Rainy Days", 
                       color="RainToday", 
                       color_discrete_sequence=px.colors.qualitative.Plotly)
    st.plotly_chart(fig)
    
    with st.expander("Afficher les Statistiques"):
        st.write(train_df.describe().transpose())

# Colonne 2 - Prédiction
with col2:
    
    # Ajouter la section explicative sur la régression logistique
    st.header("Training a Logistic Regression Model")

    st.markdown("""
    <div style="padding: 11px;">
    Logistic regression is a commonly used technique for solving binary classification problems. In a logistic regression model: 

    - we take linear combination (or weighted sum of the input features) 
    - we apply the sigmoid function to the result to obtain a number between 0 and 1
    - this number represents the probability of the input being classified as "Yes"
    - instead of RMSE, the cross entropy loss function is used to evaluate the results

    Here's a visual summary of how a logistic regression model is structured ([source](http://datahacker.rs/005-pytorch-logistic-regression-in-pytorch/)):

    <img src="https://i.imgur.com/YMaMo5D.png" style="width: 300px;">

    The sigmoid function applied to the linear combination of inputs has the following formula:

    <img src="https://i.imgur.com/sAVwvZP.png" style="width: 300px;">

    To train a logistic regression model, we can use the LogisticRegression class from Scikit-learn.
    </div>
    """, unsafe_allow_html=True)
    
    
    st.markdown("""
    ### Exemple de données d'entrée :
    
    ```python
    new_input = {
        'Date': '2021-06-19',
        'Location': 'Katherine',
        'MinTemp': 23.2,
        'MaxTemp': 33.2,
        'Rainfall': 10.2,
        'Evaporation': 4.2,
        'Sunshine': np.nan,
        'WindGustDir': 'NNW',
        'WindGustSpeed': 52.0,
        'WindDir9am': 'NW',
        'WindDir3pm': 'NNE',
        'WindSpeed9am': 13.0,
        'WindSpeed3pm': 20.0,
        'Humidity9am': 89.0,
        'Humidity3pm': 58.0,
        'Pressure9am': 1004.8,
        'Pressure3pm': 1001.5,
        'Cloud9am': 8.0,
        'Cloud3pm': 5.0,
        'Temp9am': 25.7,
        'Temp3pm': 33.0,
        'RainToday': 'Yes'
    }
    ```
    
    Vous pouvez utiliser ces valeurs comme exemple pour tester les prédictions.
    """)
    
    st.header("Raining Prediction")
    
    with st.expander("Saisir les caractéristiques"):
        input_data = {}
        
        # Ajouter le sélecteur de date
        input_data['Date'] = st.date_input('Date', value=datetime.today()).strftime('%Y-%m-%d')
        
        for col in input_cols:
            if col == 'Date':
                continue
            if col in numeric_cols:
                input_data[col] = st.number_input(col, value=0.0)
            elif col in categorical_cols:
                unique_values = aussie_rain['encoder'].categories_[categorical_cols.index(col)]
                input_data[col] = st.selectbox(col, unique_values)
            else:
                input_data[col] = st.text_input(col, value='')
    
    if st.button("Predict"):
    
        pred, prob = predict_input(input_data)
        st.write(f"La prédiction est : **{'Oui' if pred == 'Yes' else 'Non'}** avec une probabilité de {prob:.2f}")

        # Affichage de la matrice de confusion
        st.header("Matrice de Confusion")
        y_true = train_df[target_col]
        y_pred = model.predict(train_df[numeric_cols + encoded_cols])
        
        cf_matrix = confusion_matrix(y_true, y_pred, normalize='true')
        fig = px.imshow(cf_matrix, text_auto=True, labels=dict(x='Prédiction', y='Réel', color='Proportion'),
                        x=['Non', 'Oui'], y=['Non', 'Oui'], title='Matrice de Confusion',
                        color_continuous_scale=px.colors.sequential.Magma)
        st.plotly_chart(fig)

        # Affichage du rapport de classification
        st.header("Classification Report")
        report = classification_report(y_true, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.write(report_df)
        
        # Calculer les scores de probabilité pour les données d'entraînement
        y_true_binary = y_true.replace({'No': 0, 'Yes': 1})
        y_proba = model.predict_proba(train_df[numeric_cols + encoded_cols])[:, 1]

        # Compute ROC curve
        fpr, tpr, thresholds = roc_curve(y_true_binary, y_proba)

        # Plot ROC curve
        fig = px.area(x=fpr, y=tpr, title='Courbe ROC',
                    labels=dict(x='Taux de Faux Positifs', y='Taux de Vrais Positifs'),
                    width=600, height=600)
        fig.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        fig.update_xaxes(constrain='domain')
        st.plotly_chart(fig)

        # Feature Importance
        st.header("Feature Importance")
        feature_importance = model.coef_[0]
        feature_df = pd.DataFrame({
            'Feature': numeric_cols + encoded_cols,
            'Importance': feature_importance
        }).sort_values(by='Importance', ascending=False)
        fig = px.bar(feature_df, x='Importance', y='Feature', orientation='h', title='Importance des Caractéristiques')
        st.plotly_chart(fig)
