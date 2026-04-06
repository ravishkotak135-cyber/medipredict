from flask import Flask, render_template, jsonify, request, session
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import os


# ===== USER STORAGE =====
USER_FILE = "users.csv"

if not os.path.exists(USER_FILE):
    pd.DataFrame(columns=["name", "email", "password", "role"]).to_csv(USER_FILE, index=False)

print("=" * 60)
print("  MediPredict — Loading BRFSS 2015 Diabetes Dataset")
print("=" * 60)

# ─── Load & prepare dataset ───────────────────────────────────────────────────
df = pd.read_csv("diabetes_012_health_indicators_BRFSS2015.csv")
print(f"  Dataset loaded: {len(df):,} records, {len(df.columns)} features")

FEATURE_COLS = [
    'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke',
    'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
    'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth',
    'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education', 'Income'
]

X = df[FEATURE_COLS]
y = df['Diabetes_012'].apply(lambda v: 1 if v > 0 else 0)

# Stratified sample for faster startup
X_sample, _, y_sample, _ = train_test_split(X, y, train_size=100_000, random_state=42, stratify=y)
X_train, X_test, y_train, y_test = train_test_split(X_sample, y_sample, test_size=0.2, random_state=42, stratify=y_sample)

print("  Training Random Forest model (100 trees)...")
model = RandomForestClassifier(n_estimators=100, max_depth=12, min_samples_split=10, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

accuracy = round(accuracy_score(y_test, model.predict(X_test)) * 100, 2)
print(f"  Model accuracy: {accuracy}%")

FEATURE_IMPORTANCES = sorted(zip(FEATURE_COLS, model.feature_importances_), key=lambda x: x[1], reverse=True)

AGE_LABELS_MAP = {
    1:'18-24', 2:'25-29', 3:'30-34', 4:'35-39', 5:'40-44',
    6:'45-49', 7:'50-54', 8:'55-59', 9:'60-64', 10:'65-69',
    11:'70-74', 12:'75-79', 13:'80+'
}

df_sample = df.sample(5000, random_state=42).reset_index(drop=True)
df_sample['outcome'] = df_sample['Diabetes_012'].apply(lambda v: 1 if v > 0 else 0)
positive_df = df_sample[df_sample['outcome'] == 1]
negative_df = df_sample[df_sample['outcome'] == 0]

print("  Ready! Visit http://localhost:5000\n")

# ─── Flask App ────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = 'medipredict_secret_key_2024'


@app.route('/')
def index():
    return render_template('index.html', model_accuracy=accuracy)

@app.route('/api/signup', methods=['POST'])
def signup():
    data = request.get_json()

    name = data.get('name')
    email = data.get('email')
    password = data.get('password')
    role = data.get('role')

    df_users = pd.read_csv(USER_FILE)

    # Check if user already exists
    if email in df_users['email'].values:
        return jsonify({'success': False, 'message': 'User already exists'})

    # Create new user
    new_user = pd.DataFrame([{
        "name": name,
        "email": email,
        "password": password,
        "role": role
    }])

    df_users = pd.concat([df_users, new_user], ignore_index=True)
    df_users.to_csv(USER_FILE, index=False)

    return jsonify({'success': True})

@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()

    email = data.get('email')
    password = data.get('password')

    df_users = pd.read_csv(USER_FILE)

    user = df_users[
        (df_users['email'] == email) &
        (df_users['password'] == password)
    ]

    if user.empty:
        return jsonify({'success': False, 'message': 'Invalid credentials'})

    user_data = user.iloc[0].to_dict()
    session['user'] = user_data

    return jsonify({'success': True, 'user': user_data})


@app.route('/api/logout', methods=['POST'])
def do_logout():
    session.clear()
    return jsonify({'success': True})


@app.route('/api/stats')
def get_stats():
    total = len(df)
    pos_count = int((df['Diabetes_012'] > 0).sum())
    neg_count = total - pos_count
    pos_df = df[df['Diabetes_012'] > 0]
    neg_df = df[df['Diabetes_012'] == 0]

    def ca(frame, col): return round(float(frame[col].mean()), 1)

    return jsonify({
        'total': total,
        'positive': pos_count,
        'negative': neg_count,
        'prevalence': round(pos_count / total * 100, 1),
        'avg_bmi': ca(df, 'BMI'),
        'model_accuracy': accuracy,
        'diabetic':     {'avg_bmi': ca(pos_df,'BMI'), 'avg_age': ca(pos_df,'Age'), 'high_bp_pct': round(float(pos_df['HighBP'].mean())*100,1), 'high_chol_pct': round(float(pos_df['HighChol'].mean())*100,1)},
        'non_diabetic': {'avg_bmi': ca(neg_df,'BMI'), 'avg_age': ca(neg_df,'Age'), 'high_bp_pct': round(float(neg_df['HighBP'].mean())*100,1), 'high_chol_pct': round(float(neg_df['HighChol'].mean())*100,1)},
    })


@app.route('/api/patients')
def get_patients():
    page = int(request.args.get('page', 1))
    per_page = int(request.args.get('per_page', 20))
    start = (page - 1) * per_page
    slice_df = df.iloc[start:start+per_page]
    patients = []
    for i, (_, row) in enumerate(slice_df.iterrows(), start=start):
        ov = int(row['Diabetes_012'])
        patients.append({
            'id': f'#{str(i+1).zfill(5)}',
            'age': AGE_LABELS_MAP.get(int(row['Age']), str(int(row['Age']))),
            'bmi': row['BMI'],
            'high_bp': 'Yes' if row['HighBP']==1 else 'No',
            'high_chol': 'Yes' if row['HighChol']==1 else 'No',
            'smoker': 'Yes' if row['Smoker']==1 else 'No',
            'phys_activity': 'Yes' if row['PhysActivity']==1 else 'No',
            'gen_hlth': int(row['GenHlth']),
            'sex': 'Male' if row['Sex']==1 else 'Female',
            'outcome': 'Negative' if ov==0 else ('Pre-diabetic' if ov==1 else 'Diabetic'),
            'outcome_code': ov,
        })
    return jsonify({'patients': patients, 'total': len(df), 'page': page, 'per_page': per_page, 'total_pages': (len(df)+per_page-1)//per_page})


@app.route('/api/chart-data')
def get_chart_data():
    def pct(series): return round(float(series.mean())*100, 1)

    bmi_bins  = [12,18,22,26,30,35,40,50,99]
    bmi_labels = ['<18','18-22','22-26','26-30','30-35','35-40','40-50','50+']
    def bmi_bin(frame):
        return [int(((frame['BMI']>=bmi_bins[i])&(frame['BMI']<bmi_bins[i+1])).sum()) for i in range(len(bmi_bins)-1)]

    age_groups = list(range(1,14))
    age_counts = [int((df_sample['Age']==a).sum()) for a in age_groups]
    age_rate   = [round(float(df_sample[df_sample['Age']==a]['outcome'].mean())*100,1) if len(df_sample[df_sample['Age']==a])>0 else 0 for a in age_groups]
    age_bmi    = [round(float(df_sample[df_sample['Age']==a]['BMI'].mean()),1) if len(df_sample[df_sample['Age']==a])>0 else 0 for a in age_groups]

    gh_labels = ['Excellent','Very Good','Good','Fair','Poor']
    gh_pos = [int((positive_df['GenHlth']==i).sum()) for i in range(1,6)]
    gh_neg = [int((negative_df['GenHlth']==i).sum()) for i in range(1,6)]

    risk_cols   = ['HighBP','HighChol','Smoker','Stroke','HeartDiseaseorAttack','PhysActivity','DiffWalk']
    risk_labels = ['High BP','High Chol','Smoker','Stroke','Heart Disease','Phys Activity','Diff Walking']

    top8 = FEATURE_IMPORTANCES[:8]

    radar_cols   = ['HighBP','HighChol','BMI','Smoker','PhysActivity','GenHlth','Age','Income']
    radar_labels = ['High BP','High Chol','BMI','Smoker','Phys Active','Gen Health','Age','Income']
    pos_avg = [float(positive_df[c].mean()) for c in radar_cols]
    neg_avg = [float(negative_df[c].mean()) for c in radar_cols]
    mx = [max(pos_avg[i],neg_avg[i]) for i in range(len(radar_cols))]
    pos_norm = [round(pos_avg[i]/mx[i]*100,1) if mx[i] else 0 for i in range(len(radar_cols))]
    neg_norm = [round(neg_avg[i]/mx[i]*100,1) if mx[i] else 0 for i in range(len(radar_cols))]

    return jsonify({
        'outcome':      {'positive': len(positive_df), 'negative': len(negative_df)},
        'bmi':          {'labels': bmi_labels, 'positive': bmi_bin(positive_df), 'negative': bmi_bin(negative_df)},
        'age_groups':   {'labels': [AGE_LABELS_MAP[a] for a in age_groups], 'counts': age_counts, 'rate': age_rate, 'bmi': age_bmi},
        'genhlth':      {'labels': gh_labels, 'positive': gh_pos, 'negative': gh_neg},
        'risk_factors': {'labels': risk_labels, 'positive': [pct(positive_df[c]) for c in risk_cols], 'negative': [pct(negative_df[c]) for c in risk_cols]},
        'correlation':  {'labels': [f[0] for f in top8], 'values': [round(float(f[1]),4) for f in top8]},
        'radar':        {'labels': radar_labels, 'positive': pos_norm, 'negative': neg_norm},
    })


@app.route('/api/predict', methods=['POST'])
def predict():
    if 'user' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    try:
        data = request.get_json()
        input_df = pd.DataFrame([{col: float(data.get(col, 0)) for col in FEATURE_COLS}])[FEATURE_COLS]
        prediction = int(model.predict(input_df)[0])
        probability = round(float(model.predict_proba(input_df)[0][1]) * 100, 1)

        if probability < 25:   risk_level, risk_color = 'Low Risk',       'green'
        elif probability < 50: risk_level, risk_color = 'Moderate Risk',  'amber'
        elif probability < 75: risk_level, risk_color = 'High Risk',      'orange'
        else:                  risk_level, risk_color = 'Very High Risk',  'red'

        fi = dict(zip(FEATURE_COLS, model.feature_importances_))
        top_factors = [
            {'feature': f, 'importance': round(imp*100,1), 'value': float(data.get(f,0))}
            for f, imp in sorted(fi.items(), key=lambda x: x[1], reverse=True)[:5]
        ]
        return jsonify({'prediction': prediction, 'probability': probability, 'risk_level': risk_level,
                        'risk_color': risk_color, 'result': 'Diabetic / Pre-Diabetic' if prediction==1 else 'Non-Diabetic',
                        'top_factors': top_factors, 'model_accuracy': accuracy})
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 500


import random

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.get_json()
    message = data.get("message", "").lower()

    if len(message.split()) > 6 and not any(word in message for word in ["diabetes", "health", "bmi", "symptoms"]):
        reply = "That seems like a general question. I specialize in medical assistance. Try asking about health or diabetes."

    # Greeting
    if any(word in message for word in ["hi", "hello", "hey"]):
        reply = random.choice([
            "Hello! How can I assist you today?",
            "Hi there! Ask me anything about diabetes or health.",
            "Hey! I'm your AI medical assistant."
        ])

    # Diabetes
    elif any(word in message for word in ["diabetes", "sugar", "blood sugar"]):
        reply = "Diabetes is a condition where the body cannot properly regulate blood sugar levels."

    # Symptoms
    elif any(word in message for word in ["symptoms", "signs", "feel"]):
        reply = "Common symptoms include frequent urination, increased thirst, fatigue, and blurred vision."

    # Prevention
    elif any(word in message for word in ["prevent", "avoid", "reduce"]):
        reply = "You can reduce the risk by exercising regularly, eating a balanced diet, and maintaining healthy weight."

    # Diet
    elif any(word in message for word in ["diet", "food", "eat"]):
        reply = "A healthy diet includes low sugar, high fiber foods, vegetables, and whole grains."

    # Exercise
    elif any(word in message for word in ["exercise", "workout", "gym"]):
        reply = "Regular physical activity helps control blood sugar and improves overall health."

    # BMI
    elif "bmi" in message:
        reply = "BMI (Body Mass Index) measures body fat based on height and weight."

    # Prediction
    elif any(word in message for word in ["prediction", "result", "analysis"]):
        reply = "Your prediction is based on factors like BMI, blood pressure, and lifestyle habits."

    elif any(word in message for word in ["health", "healthy", "wellbeing"]):
        reply = "Maintaining good health involves balanced diet, regular exercise, proper sleep, and stress management."

    elif "sleep" in message:
        reply = "Good sleep helps regulate hormones and improves overall health. Aim for 7-8 hours daily."

    elif "water" in message:
        reply = "Drinking enough water helps maintain blood sugar balance and overall body function."

    elif "stress" in message:
        reply = "Stress can increase blood sugar levels. Try meditation, exercise, and relaxation techniques."

    elif "type 1" in message:
        reply = "Type 1 diabetes occurs when the body does not produce insulin."

    elif "type 2" in message:
        reply = "Type 2 diabetes occurs when the body becomes resistant to insulin."

    elif "insulin" in message:
        reply = "Insulin is a hormone that helps regulate blood sugar levels."

    elif "glucose" in message:
        reply = "Glucose is the main source of energy but too much can be harmful."

    elif "fasting sugar" in message:
        reply = "Normal fasting blood sugar is usually between 70–100 mg/dL."

    elif "thirst" in message:
        reply = "Excessive thirst is a common symptom of high blood sugar."

    elif "urination" in message:
        reply = "Frequent urination can indicate high glucose levels."

    elif "fatigue" in message:
        reply = "Fatigue occurs when cells cannot use glucose efficiently."

    elif "blurred vision" in message:
        reply = "High sugar levels can affect eye lenses causing blurred vision."

    elif "sugar" in message:
        reply = "Limiting sugar intake helps control blood glucose levels."

    elif "fruits" in message:
        reply = "Fruits are healthy but should be consumed in moderation due to natural sugars."

    elif "vegetables" in message:
        reply = "Vegetables are rich in fiber and help control blood sugar."

    elif "junk" in message:
        reply = "Avoid junk food as it contains high sugar and unhealthy fats."

    elif "carbs" in message:
        reply = "Carbohydrates affect blood sugar levels, so they should be consumed wisely."

    elif "walking" in message:
        reply = "Walking daily for 30 minutes can significantly improve health."

    elif "running" in message:
        reply = "Running helps burn calories and improves insulin sensitivity."

    elif "gym" in message:
        reply = "Regular gym workouts help maintain weight and reduce diabetes risk."

    elif "yoga" in message:
        reply = "Yoga helps reduce stress and improve metabolic health."

    elif "risk" in message:
        reply = "Risk factors include obesity, genetics, lack of exercise, and unhealthy diet."

    elif "family history" in message:
        reply = "Having a family history increases diabetes risk."

    elif "obesity" in message:
        reply = "Obesity is one of the leading causes of type 2 diabetes."

    elif "weight" in message:
        reply = "Maintaining a healthy weight helps reduce diabetes risk."

    elif "overweight" in message:
        reply = "Being overweight increases insulin resistance."

    elif "height" in message:
        reply = "Height is used along with weight to calculate BMI."

    elif "accuracy" in message:
        reply = "The model accuracy represents how well the prediction matches real outcomes."

    elif "model" in message:
        reply = "The prediction uses a Random Forest machine learning model."

    elif "factors" in message:
        reply = "Factors like BMI, age, blood pressure, and lifestyle influence predictions."

    elif "lifestyle" in message:
        reply = "Healthy lifestyle includes balanced diet, exercise, and avoiding smoking."

    elif "smoking" in message:
        reply = "Smoking increases the risk of diabetes and heart disease."

    elif "alcohol" in message:
        reply = "Excess alcohol can increase blood sugar levels."

    elif "routine" in message:
        reply = "A healthy routine includes exercise, proper meals, and regular sleep."

    elif "morning" in message:
        reply = "Start your day with light exercise and a healthy breakfast."

    elif "night" in message:
        reply = "Avoid heavy meals at night and ensure proper sleep."

    elif "who are you" in message:
        reply = "I am your AI medical assistant designed to help with health insights."

    elif "what can you do" in message:
        reply = "I can explain diabetes, give health tips, and help you understand predictions."

    elif "help" in message:
        reply = "You can ask me about diabetes, symptoms, diet, exercise, or predictions."

    elif any(word in message for word in ["mountain", "earth", "capital", "who", "what", "where", "when"]):
        reply = "I specialize in medical and diabetes-related queries. For general knowledge questions, please ask something related to health."


    else:

        reply = random.choice([

            "I’m focused on healthcare and diabetes. Try asking about symptoms, diet, or prediction results.",

            "That’s outside my medical expertise. Ask me something related to health or diabetes.",

            "I can help with medical insights and predictions. What would you like to know?"

        ])


    return jsonify({"reply": reply})



if __name__ == '__main__':
    app.run(debug=True, port=5000)
