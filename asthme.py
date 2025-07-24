import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from PIL import Image
import base64
import pickle
import io
import requests


@st.cache_data
def load_data(dataset):
    df = pd.read_csv(dataset)
    return df

def file_download(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="BMI_IOS_SCD_Asthma.csv">Download CSV File</a>'
    return href


def predict_asthma(Hydroxyurea, ICS, LABA, Gender, Age, Height , Weight, BMI, R5Hz_PP, R20Hz_PP, X5Hz_PP, Fres_PP):
    try:
        with open('nettoyage/model_asthma.pkl', 'rb') as file:
                        loaded_model = pickle.load(file)

        nouvelles_donnees = pd.DataFrame({
            "Hydroxyurea": [Hydroxyurea],
            "ICS": [ICS],
            "LABA": [LABA],
            "Gender": [Gender],
            "Age": [Age],
            "Height": [Height],
            "Weight": [Weight],
            "BMI": [BMI],
            "R5Hz_PP": [R5Hz_PP],
            "R20Hz_PP": [R20Hz_PP],
            "X5Hz_PP": [X5Hz_PP],
            "Fres_PP": [Fres_PP]
        })

        prediction = loaded_model.predict(nouvelles_donnees)
        if prediction[0] == 0:
            return "✅ No Asthma"
        else:
            return "⭕️ Asthma"
    except FileNotFoundError:
        st.error("❌ File of Machin Learning Model Unfounded")
    except Exception as e:
        st.error(f'⚠️ Error into the prediction {str(e)}')
    


def load_local_css(file_path):
    with open(file_path, "r") as f:
        css_content = f.read()
        st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)

load_local_css("Folder_style/asthma.css")

def main():
    # Initialiser la page active si elle n'existe pas encore
    if 'page' not in st.session_state:
        st.session_state.page = "Home"

    # Sidebar image path correction
    st.sidebar.image("im_pr/asthi-check.png", width=300)

    # Boutons du menu sidebar
    if st.sidebar.button("Home"):
        st.session_state.page = "Home"
    if st.sidebar.button("Visualisation"):
        st.session_state.page = "Visualisation"
    if st.sidebar.button("Analyse"):
        st.session_state.page = "Analyse"
    if st.sidebar.button("ChatBot"):
        st.session_state.page = "ChatBot"

    # Utiliser st.session_state.page comme choix
    choice = st.session_state.page

    if choice not in ["Visualisation", "Analyse", "ChatBot"]:
        st.title("Bienvenue sur ASTHI-CHECK")
    
    
    # Load dataset
    data = load_data("Dataset/BMI_IOS_SCD_Asthma.csv")
    
    if choice == "Home":
        st.subheader("Qu'est-ce que l'asthme ?")

        xtab1,xtab2 = st.columns(2)
        with xtab1:
            st.write("L’asthme est une affection respiratoire chronique qui touche des millions de personnes à travers le monde entier." \
                    " Cette maladie se caractérise par une inflammation, mais aussi un rétrécissement des voies respiratoires, ce qui rend la respiration assez difficile." \
                    " Les symptômes de l’asthme peuvent inclure une respiration sifflante, un essoufflement, une sensation d’oppression thoracique" \
                    " et une toux, en particulier la nuit ou tôt le matin. Bien que l’asthme ne puisse pas être guéri, l'on peut gérer cette condition de manière efficace avec un traitement approprié et des changements de mode de vie." \
                    " Les personnes asthmatiques peuvent avoir une vie normale. Cela en évitant les déclencheurs, en prenant des médicaments prescrits et en suivant les conseils de leur médecin." \
                    " Alors la sensibilisation à l’asthme et à ses symptômes est essentielle pour améliorer la qualité de vie des personnes atteintes et pour prévenir les crises d’asthme graves.")
        
        with xtab2:
            st.image("im_pr/asthme.jpeg", width=400)
            st.image("im_pr/asthme11.jpg", width=400)


    elif choice == "Visualisation":
        st.title("Visualisation des données")

        asthma_filter = st.selectbox("Choisissez le statut de la maladie", pd.unique(data['Asthma']))
        data = data[data['Asthma'] == asthma_filter]

        avg_r5 = np.mean(data['R5Hz_PP'])
        count_gender= int(data[(data['Gender'] == 'Male')]['Gender'].count())
        avg_r20 = np.mean(data['R20Hz_PP'])
        avg_fres = np.mean(data['Fres_PP'])

        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric(label="Moyenne de R5Hz_PP 🩸", value=round(avg_r5), delta=round(avg_r5))
        kpi2.metric(label="Compte par genre 🚹/🚺", value=count_gender, delta=round(count_gender))
        kpi3.metric(label="Moyenne de R20Hz_PP 💉", value=f'{round(avg_r20,2)}', delta=f'{round(avg_r20,2)}')
        kpi4.metric(label="Moyenne de Fres_PP 💊", value=round(avg_fres), delta=round(avg_fres))

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Count Plot du top 10 des valeurs du BMI")
            fig1 = plt.figure(figsize=(9, 5.97))
            top_values = data['BMI'].value_counts().nlargest(10).index
            filtered_data = data[data['BMI'].isin(top_values)]
            sns.countplot(data=filtered_data, x="BMI", palette='muted')
            plt.title('Count Plot des 10 premiers BMI')
            st.pyplot(fig1)


            st.subheader("Scatter Plot des tailles en fonction des poids")
            fig2 = plt.figure(figsize=(9, 7.5))
            sns.scatterplot(x="Height (cm)", y="Weight (Kg)", data=data)
            plt.title("Scatter Plot des tailles en fonction des poids")
            st.pyplot(fig2)
        
        with col2:
            st.subheader("Pie Chart du top 4 des valeurs fréquentes")
            numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
            numeric_cols = [col for col in numeric_cols if col != 'Unnamed: 0']

            if len(numeric_cols) > 0:
                selected_col = st.selectbox("Sélectionnez une constante", ['R5Hz_PP', 'R20Hz_PP', 'Fres_PP'])
                top_values = data[selected_col].value_counts().nlargest(4)
                fig3 = plt.figure(figsize=(2,2.2))
                plt.pie(top_values.values, labels=top_values.index, autopct='%1.1f%%', startangle=90)
                plt.title(f'Distribution du top 4 de valeurs fréquentes {selected_col}')
                plt.axis("equal")
                st.pyplot(fig3)
            
            st.subheader('Boxplot du genre en fonction de l’âge')
            data_filtered = data[data['Gender'] != 'male']
            fig4 = plt.figure(figsize=(5,4.35))
            sns.boxplot(x="Gender", y="Age (months)", data=data_filtered, palette='viridis')
            st.pyplot(fig4)
            
    elif choice == "Analyse":
        st.title("Analyse")
        tab1,tab2 = st.tabs([":microscope: Machine Learning", ":brain: Deep Learning"])
        
        with tab1:
            # Sliders pour les valeurs numériques
            st.subheader("Formulaire des paramètres asthmatiques")

            st.write("Entrez les paramètres démographiques :")
            stab1, stab2 = st.columns(2)

            with stab1:
                    Age = st.number_input("Age", min_value=0.0, max_value=100.0, value=0.0, step=1.0)
                    Height = st.number_input("Height", min_value=0, max_value=230, value=0, step=1)
                    Gender = st.selectbox("Gender", options=["0", "1"])

            with stab2:
                    BMI = st.number_input("BMI", min_value=0.0, max_value=40.0, value=0.0, step=1.0)
                    Weight = st.number_input("Weight", min_value=0, max_value=150, value=0, step=1)


            st.write("Entrez les paramètres oscillométriques :")
            sstab1, sstab2 = st.columns(2)
            with sstab1:
                    R5Hz_PP = st.number_input("R5Hz_PP", min_value=0.0, max_value=200.0, value=0.0, step=1.0)
                    R20Hz_PP = st.number_input("R20Hz_PP", min_value=0.0, max_value=160.0, value=0.0, step=1.0)

            with sstab2:
                    X5Hz_PP = st.number_input("X5Hz_PP", min_value=-100.0, max_value=250.0, value=0.0, step=1.0)
                    Fres_PP = st.number_input("Fres_PP", min_value=0.0, max_value=230.0, value=0.0, step=1.0)
                    
                    
            st.write("Entrez les autres paramètres :")
            ssstab1, ssstab2 = st.columns(2)
            with ssstab1:
                LABA = st.selectbox("LABA", options=["0", "1"])
                ICS = st.selectbox("ICS", options=["0", "1"])
            
            with ssstab2:
                Hydroxyurea = st.selectbox("Hydroxyurea", options=["0", "1"])


            button = st.button("Prediction")
            if button:
                New_Asthma = predict_asthma(Hydroxyurea, ICS, LABA, Gender, Age, Height , Weight, BMI, R5Hz_PP, R20Hz_PP, X5Hz_PP, Fres_PP)
                if New_Asthma:
                    st.success(f"The patient have : **{New_Asthma}**")


                    # Sauvegarde des données dans session_state pour le ChatBot
                    st.session_state['derniere_prediction'] = {
                        "Hydroxyurea": Hydroxyurea,
                        "ICS": ICS,
                        "LABA": LABA,
                        "Gender": Gender,
                        "Age": Age,
                        "Height": Height,
                        "Weight": Weight,
                        "BMI": BMI,
                        "R5Hz_PP": R5Hz_PP,
                        "R20Hz_PP": R20Hz_PP,
                        "X5Hz_PP": X5Hz_PP,
                        "Fres_PP": Fres_PP,
                        "Résultat": New_Asthma
                    }

        with tab2:
            
            deep_model = load_model('nettoyage/deep_asthma.h5')

            image_height = 200
            image_width = 200

            uploaded_file = st.file_uploader("Sélectionnez une image médicale pour l'analyse par imagerie", type=["jpg", "jpeg", "png"])

            if uploaded_file is not None:

                image = Image.open(uploaded_file).convert('RGB')
                st.image(image, caption='Image sélectionnée', use_container_width=True)

                # Prétraitement
                img = image.resize((image_height, image_width))
                img_array = np.array(img) / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                prediction = deep_model.predict(img_array)

                if prediction[0][0] > 0.5:
                    result_dl = "Asthmatique"
                else:
                    result_dl = "Normale"

                st.success(f"🫁 Résultat Deep Learning : {result_dl}")

                # Sauvegarde dans session_state pour le ChatBot
                st.session_state['derniere_prediction_dl'] = {
                    "Résultat_DL": result_dl
                }


    elif choice == "ChatBot":
        col1, col2 = st.columns([0.06, 0.94])  # Ajuste les proportions selon la taille de l’image
        with col1:
            st.image("im_pr/chatbot.png", width=40)

        with col2:
            st.markdown('<h2 style="margin: 0; padding: 0; color:#06d9e0">Discuss with ASTHIBOT<h2>', unsafe_allow_html=True)

            st.markdown("""
            <style>
            .typewriter-container {
                text-align: center;
                margin-top: 10px;
                margin-bottom: 1px;
                font-family: Germania One, sans-serif;
                font-size: 20px;
                color: #ffffff;
            }

            .typewriter-box {
                display: inline-block;
                background-color: #000000; /* couleur de fond du cadre */
                color: white; /* couleur du texte à l'intérieur */
                padding: 10px 14px; /* espace interne */
                border: 2px solid #2a2a2a; /* bordure du cadre */
            }

            .typewriter-text {
                display: inline-block;
                overflow: hidden;
                white-space: nowrap;
                animation:
                    typing 3s steps(50, end),
                    fadeBlur 1.5s ease-out forwards;
                filter: blur(6px);
            }

            @keyframes typing {
                from { width: 0 }
                to { width: 100% }
            }

            @keyframes fadeBlur {
                0% { filter: blur(6px); }
                100% { filter: blur(0px); }
            }
            </style>

            <div class="typewriter-container">
                <div class="typewriter-box">
                    <div class="typewriter-text">👋 Bonjour, je suis AsthiBot ! Comment puis-je vous aider aujourd'hui ?</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Liste des mots-clés autorisés pour filtrer les questions
        allowed_keywords = [
            "asthme", "asthma", "symptôme", "symptoms", "traitement", "treatment",
            "R5Hz", "R20Hz", "X5Hz", "Fres", "BMI", "ICS", "LABA", "Hydroxyurea",
            "prédiction", "prediction", "résultat", "result", "diagnostic", "diagnosis",
            "proposition", "proposition de traitement", "question", "réponse",
            "répondre", "answer", "aide", "help", "conseil", "advice", "informations", "information",
            "image", "radiographie", "scanner", "état de santé", "health status"
        ]

        def is_question_relevant(question):
            return any(keyword.lower() in question.lower() for keyword in allowed_keywords)

        API_URL = "https://router.huggingface.co/nebius/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {st.secrets['Token1']}",
        }

        def query_hf_model(message, prediction_data=None, prediction_data_dl=None):

            # Construction du contexte avec les dernières données de prédiction
            context = "Tu es un expert en santé respiratoire. Réponds uniquement aux questions concernant " \
                    "l'asthme ou les résultats de prédiction liés à l'asthme. Donne des réponses détaillées, " \
                    "claires et faciles à comprendre pour un patient ou un professionnel de santé."
            if prediction_data:
                context += f"""
            Voici les dernières données du patient à considérer pour le contexte :
            Hydroxyurea: {prediction_data.get("Hydroxyurea", "N/A")}
            ICS: {prediction_data.get("ICS", "N/A")}
            LABA: {prediction_data.get("LABA", "N/A")}
            Gender: {prediction_data.get("Gender", "N/A")}
            Age: {prediction_data.get("Age", "N/A")}
            Height: {prediction_data.get("Height", "N/A")}
            Weight: {prediction_data.get("Weight", "N/A")}
            BMI: {prediction_data.get("BMI", "N/A")}
            R5Hz_PP: {prediction_data.get("R5Hz_PP", "N/A")}
            R20Hz_PP: {prediction_data.get("R20Hz_PP", "N/A")}
            X5Hz_PP: {prediction_data.get("X5Hz_PP", "N/A")}
            Fres_PP: {prediction_data.get("Fres_PP", "N/A")}
            Résultat de la prédiction : {prediction_data.get("Résultat", "N/A")}
            """
            
            if prediction_data_dl:
                    context += f"""
            Voici le résultat de la dernière prédiction Deep Learning :
            Résultat prédiction DL : {prediction_data_dl.get("Résultat_DL", "N/A")}
            """

            payload = {
                "messages": [
                    {"role": "system", "content": context},
                    {"role": "user", "content": message}
                ],
                "model": "microsoft/phi-4",
                "temperature": 0.4,
                "max_tokens": 768
            }
            try:
                response = requests.post(API_URL, headers=headers, json=payload)
                if response.status_code == 200:
                    return response.json()["choices"][0]["message"]["content"]
                else:
                    return f"Erreur d'appel API : {response.status_code}"
            except Exception as e:
                return f"Erreur : {str(e)}"
        

        # Récupérer les données de la dernière prédiction
        prediction_data = st.session_state.get('derniere_prediction', {})

        prediction_data_dl = st.session_state.get('derniere_prediction_dl', {})

        user_message = st.chat_input("Veuillez poser votre question (ex: Que signifie R5Hz_PP ?)")

        if user_message:
            if is_question_relevant(user_message):
                with st.spinner("ASTHIBOT réfléchit..."):
                    bot_reply = query_hf_model(user_message, prediction_data, prediction_data_dl)
                

                # Stocker la conversation
                st.session_state.messages.append({"role": "user", "content": user_message})
                st.session_state.messages.append({"role": "assistant", "content": bot_reply})
            else:
                st.warning("⚠️ Je ne peux répondre à cette question. Veuillez réessayer")
        
        # Affichage des messages avec style bulle
        for msg in st.session_state.messages:
            clean_content = msg["content"].replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
            
            if msg["role"] == "user":
                st.markdown(f"""
                    <div style='display: flex; justify-content: flex-end; margin: 10px 0;'>
                        <div style='position: relative; background-color: #f0f0f0; color: black; padding: 10px 14px; border-radius: 16px; max-width: 70%;'>
                            <div style='white-space: pre-wrap;'>{clean_content}</div>
                            <div style="
                                content: '';
                                position: absolute;
                                right: -10px;
                                top: 10px;
                                width: 0;
                                height: 0;
                                border-top: 10px solid transparent;
                                border-bottom: 10px solid transparent;
                                border-left: 10px solid #f0f0f0;
                            "></div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div style='display: flex; justify-content: flex-start; margin: 10px 0;'>
                        <div style='position: relative; background-color: #2b2a2a; color: white; padding: 10px 14px; border-radius: 16px; max-width: 70%;'>
                            <div style='white-space: pre-wrap;'>{clean_content}</div>
                            <div style="
                                content: '';
                                position: absolute;
                                left: -10px;
                                top: 10px;
                                width: 0;
                                height: 0;
                                border-top: 10px solid transparent;
                                border-bottom: 10px solid transparent;
                                border-right: 10px solid #2b2a2a;
                            "></div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()