import sqlite3
from groq import Groq
from flask import Flask, render_template, jsonify, request, session
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import os
DB_PATH = os.path.join(os.path.dirname(__file__), 'users.db')

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ===== USER STORAGE =====
def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

# Create table if not exists
conn = get_db_connection()
conn.execute('''
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    email TEXT UNIQUE,
    password TEXT,
    role TEXT
)
''')
conn.commit()
conn.close()

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

def get_ai_response(user_input):
    health_keywords = [
        "diabetes", "health", "medical", "blood", "sugar", "bp", "bmi",
        "exercise", "diet", "symptom", "disease", "insulin",
        "heart", "cholesterol", "pressure", "glucose", "patient"
    ]

    user_input_lower = user_input.lower()

    is_health_related = any(keyword in user_input_lower for keyword in health_keywords)

    non_health_words = ["who", "movie", "actor", "cricket", "player", "code", "python"]

    if not is_health_related or any(word in user_input_lower for word in non_health_words):
        return "I am designed to answer only health-related queries."
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content":
                    "You are a healthcare assistant specialized in diabetes and general health. "
                    "Only answer questions related to diabetes, health, lifestyle, symptoms, or prevention. "
                    "If the question is unrelated (like technology, movies, coding, etc.), politely refuse by saying: "
                    "'I am designed to answer only health-related queries.' "
                    "Give short answers in 3-5 bullet points. Do not diagnose or give medical prescriptions."
                 },
                {
                    "role": "user",
                    "content": user_input
                }
            ],
            model="llama-3.1-8b-instant",
        )

        return chat_completion.choices[0].message.content

    except Exception as e:
        print("🔥 ERROR:", e)   # 👈 THIS IS KEY
        return str(e)          # 👈 SHOW ACTUAL ERROR IN UI

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

    conn = get_db_connection()

    # Check if user exists
    existing = conn.execute(
        'SELECT * FROM users WHERE email = ?',
        (email,)
    ).fetchone()

    if existing:
        conn.close()
        return jsonify({'success': False, 'message': 'User already exists'})

    # Insert new user
    conn.execute(
        'INSERT INTO users (name, email, password, role) VALUES (?, ?, ?, ?)',
        (name, email, password, role)
    )

    conn.commit()
    conn.close()

    return jsonify({'success': True})


@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()

    email = data.get('email')
    password = data.get('password')

    conn = get_db_connection()

    user = conn.execute(
        'SELECT * FROM users WHERE email = ? AND password = ?',
        (email, password)
    ).fetchone()

    conn.close()

    if not user:
        return jsonify({'success': False, 'message': 'Invalid credentials'})

    user_data = dict(user)
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
        input_data = {col: float(data.get(col, 0)) for col in FEATURE_COLS}

        # validation
        input_data['MentHlth'] = max(0, min(input_data['MentHlth'], 30))
        input_data['PhysHlth'] = max(0, min(input_data['PhysHlth'], 30))

        input_df = pd.DataFrame([input_data])[FEATURE_COLS]
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

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data.get("message", "")

    ai_reply = get_ai_response(user_message)

    return jsonify({"reply": ai_reply})

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
