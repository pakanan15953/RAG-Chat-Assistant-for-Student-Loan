import streamlit as st
import pandas as pd
import sqlite3
from datetime import datetime

# ---------------------- Database ----------------------
DB_PATH = "questions.db"
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
c = conn.cursor()

# ---------------------- ฟังก์ชันดึงข้อมูล ----------------------
def get_all_messages():
    c.execute("SELECT id, user_message, answer, timestamp FROM user_messages ORDER BY id DESC")
    return c.fetchall()

def get_all_chunks():
    c.execute("""
        SELECT rc.id, rc.user_message_id, um.user_message, rc.chunk_text, rc.source, rc.page_number
        FROM retrieved_chunks rc
        LEFT JOIN user_messages um ON rc.user_message_id = um.id
        ORDER BY rc.id DESC
    """)
    return c.fetchall()

def get_latest_metrics(limit=50):
    c.execute("""
        SELECT m.id, m.user_message_id, um.user_message, m.prompt_tokens, m.response_tokens, m.response_time, m.timestamp
        FROM llm_metrics m
        LEFT JOIN user_messages um ON m.user_message_id = um.id
        ORDER BY m.id DESC
        LIMIT ?
    """, (limit,))
    return c.fetchall()

def get_all_feedback():
    c.execute("""
        SELECT f.id, f.user_message_id, um.user_message, f.satisfaction, f.feedback_text, f.timestamp
        FROM feedback f
        LEFT JOIN user_messages um ON f.user_message_id = um.id
        ORDER BY f.id DESC
    """)
    return c.fetchall()

# ---------------------- Streamlit Page Config ----------------------
st.set_page_config(
    page_title="Admin Dashboard - RAG Chatbot", 
    page_icon="🔐", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------- Custom CSS ----------------------
st.markdown("""
    <style>
    .main {
        padding: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    .metric-label {
        font-size: 1rem;
        opacity: 0.9;
    }
    .metric-delta {
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }
    h1 {
        color: #1f2937;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    h2, h3 {
        color: #374151;
        margin-top: 1.5rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 1rem 2rem;
        font-weight: 600;
    }
    .feedback-card {
        background: #f9fafb;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3b82f6;
        margin: 0.5rem 0;
    }
    .stats-box {
        background: #f0f9ff;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------- Header ----------------------
col_header1, col_header2 = st.columns([3, 1])
with col_header1:
    st.title("🔐 ระบบเจ้าหน้าที่ - RAG Chatbot Dashboard")
    st.markdown("**ระบบติดตามและวิเคราะห์การใช้งาน Chatbot แบบ Real-time**")
with col_header2:
    st.markdown(f"### 📅 {datetime.now().strftime('%d/%m/%Y')}")
    st.markdown(f"🕐 {datetime.now().strftime('%H:%M:%S')}")

st.divider()

# ---------------------- ดึงข้อมูล ----------------------
messages_data = get_all_messages()
df_messages = pd.DataFrame(messages_data, columns=["ID", "User Message", "Answer", "Timestamp"])
if not df_messages.empty:
    df_messages["Timestamp"] = pd.to_datetime(df_messages["Timestamp"])

chunks_data = get_all_chunks()
df_chunks = pd.DataFrame(chunks_data, columns=["ID", "User Message ID", "User Message", "Chunk Text", "Source", "Page Number"])

metrics_data = get_latest_metrics(limit=50)
df_metrics = pd.DataFrame(metrics_data, columns=["ID", "User Message ID", "User Message", "Prompt Tokens", "Response Tokens", "Response Time (s)", "Timestamp"])
if not df_metrics.empty:
    df_metrics["Timestamp"] = pd.to_datetime(df_metrics["Timestamp"])

feedback_data = get_all_feedback()
df_feedback = pd.DataFrame(feedback_data, columns=["ID", "User Message ID", "User Message", "Satisfaction", "Feedback Text", "Timestamp"])
if not df_feedback.empty:
    df_feedback["Timestamp"] = pd.to_datetime(df_feedback["Timestamp"])

# ---------------------- KPI Dashboard ----------------------
st.subheader("📊 Key Performance Indicators")

col1, col2, col3, col4 = st.columns(4)

with col1:
    total_messages = len(df_messages)
    st.markdown(f"""
    <div class="metric-card" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);">
        <div class="metric-label">💬 ข้อความทั้งหมด</div>
        <div class="metric-value">{total_messages}</div>
        <div class="metric-delta">+{total_messages} คำถาม</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    feedback_count = len(df_feedback)
    satisfaction_rate = round(df_feedback[df_feedback["Satisfaction"] == "พอใจ"].shape[0] / feedback_count * 100, 1) if feedback_count > 0 else 0
    st.markdown(f"""
    <div class="metric-card" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
        <div class="metric-label">⭐ Feedback</div>
        <div class="metric-value">{feedback_count}</div>
        <div class="metric-delta">{satisfaction_rate}% พอใจ</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    avg_response_time = round(df_metrics["Response Time (s)"].mean(), 2) if not df_metrics.empty else 0
    st.markdown(f"""
    <div class="metric-card" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
        <div class="metric-label">⚡ Avg Response Time</div>
        <div class="metric-value">{avg_response_time}s</div>
        <div class="metric-delta">{'ยอดเยี่ยม' if avg_response_time < 2 else 'ปานกลาง'}</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    total_tokens = df_metrics["Prompt Tokens"].sum() + df_metrics["Response Tokens"].sum() if not df_metrics.empty else 0
    st.markdown(f"""
    <div class="metric-card" style="background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);">
        <div class="metric-label">🎯 Total Tokens Used</div>
        <div class="metric-value">{total_tokens:,}</div>
        <div class="metric-delta">{df_metrics.shape[0]} requests</div>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ---------------------- Quick Stats Section ----------------------
if not df_metrics.empty:
    st.subheader("📈 Quick Statistics")
    
    stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
    
    with stat_col1:
        st.markdown("""
        <div class="stats-box">
            <h4 style="margin:0; color:#3b82f6;">Avg Prompt Tokens</h4>
            <h2 style="margin:0.5rem 0;">{}</h2>
        </div>
        """.format(f"{df_metrics['Prompt Tokens'].mean():.0f}"), unsafe_allow_html=True)
    
    with stat_col2:
        st.markdown("""
        <div class="stats-box">
            <h4 style="margin:0; color:#8b5cf6;">Avg Response Tokens</h4>
            <h2 style="margin:0.5rem 0;">{}</h2>
        </div>
        """.format(f"{df_metrics['Response Tokens'].mean():.0f}"), unsafe_allow_html=True)
    
    with stat_col3:
        st.markdown("""
        <div class="stats-box">
            <h4 style="margin:0; color:#ec4899;">Max Response Time</h4>
            <h2 style="margin:0.5rem 0;">{}</h2>
        </div>
        """.format(f"{df_metrics['Response Time (s)'].max():.2f}s"), unsafe_allow_html=True)
    
    with stat_col4:
        st.markdown("""
        <div class="stats-box">
            <h4 style="margin:0; color:#10b981;">Min Response Time</h4>
            <h2 style="margin:0.5rem 0;">{}</h2>
        </div>
        """.format(f"{df_metrics['Response Time (s)'].min():.2f}s"), unsafe_allow_html=True)
    
    st.divider()

# ---------------------- ⭐⭐⭐ START: NEW SECTION ⭐⭐⭐ ----------------------
# ---------------------- Top Questions Analysis ----------------------
st.subheader("💡 คำถามที่พบบ่อยที่สุด (Top 10)")

if not df_messages.empty:
    # --- START: TEXT NORMALIZATION ---
    
    # สร้างสำเนาของ DataFrame เพื่อไม่ให้กระทบกับข้อมูลดิบที่แสดงในตารางอื่น
    analysis_df = df_messages.copy()

    # สร้างฟังก์ชันสำหรับทำความสะอาดข้อความ
    def normalize_text(text):
        if not isinstance(text, str):
            return ""
        text = text.lower() # 1. แปลงเป็นตัวพิมพ์เล็ก
        text = text.strip() # 2. ตัดช่องว่างหน้า-หลัง
        
        # 3. ลบคำลงท้ายและเครื่องหมายต่างๆ ที่ไม่สำคัญ
        words_to_remove = ['คะ', 'ครับ', 'ค่ะ', 'คับ', 'จ้ะ', 'จ้า', '?', '!', '.']
        for word in words_to_remove:
            text = text.replace(word, '')
            
        return text.strip() # ตัดช่องว่างอีกครั้งหลังลบคำ

    # สร้างคอลัมน์ใหม่สำหรับข้อความที่ทำความสะอาดแล้ว
    analysis_df['Normalized Message'] = analysis_df['User Message'].apply(normalize_text)

    # --- END: TEXT NORMALIZATION ---

    # 1. นับความถี่จากคอลัมน์ที่ทำความสะอาดแล้ว
    top_questions_df = analysis_df['Normalized Message'].value_counts().reset_index()
    top_questions_df.columns = ['Question', 'Count']
    
    # 2. เตรียมข้อมูลสำหรับแสดงผล
    top_10_questions = top_questions_df.head(10)

    # ... ส่วนที่เหลือของโค้ด (st.columns, st.bar_chart, st.dataframe) เหมือนเดิม ...
    col_chart, col_table = st.columns([2, 3]) 

    with col_chart:
        st.markdown("##### กราฟแสดง 10 อันดับคำถาม")
        chart_data = top_10_questions.set_index('Question')
        st.bar_chart(chart_data)

    with col_table:
        st.markdown("##### ตารางข้อมูล")
        st.dataframe(
            top_10_questions,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Question": st.column_config.TextColumn("คำถาม", width="large"),
                "Count": st.column_config.NumberColumn("จำนวนครั้ง", width="small")
            }
        )
else:
    st.info("ยังไม่มีข้อมูลเพียงพอที่จะวิเคราะห์คำถามที่พบบ่อย")

st.divider()

# ---------------------- ⭐⭐⭐ END: NEW SECTION ⭐⭐⭐ ----------------------


# ---------------------- Tabs for Data Tables ----------------------
tab1, tab2, tab3, tab4 = st.tabs(["💬 ข้อความทั้งหมด", "📂 Retrieved Chunks", "📊 LLM Metrics", "⭐ Feedback"])

with tab1:
    st.markdown("### 📋 ข้อความและคำตอบทั้งหมด")
    if not df_messages.empty:
        # Search/Filter
        search_term = st.text_input("🔍 ค้นหาคำถาม", placeholder="พิมพ์คำค้นหา...")
        if search_term:
            filtered_df = df_messages[df_messages["User Message"].str.contains(search_term, case=False, na=False)]
        else:
            filtered_df = df_messages
        
        st.dataframe(
            filtered_df,
            use_container_width=True,
            height=400,
            column_config={
                "ID": st.column_config.NumberColumn("ID", width="small"),
                "Timestamp": st.column_config.DatetimeColumn("เวลา", format="DD/MM/YYYY HH:mm"),
                "User Message": st.column_config.TextColumn("คำถาม", width="medium"),
                "Answer": st.column_config.TextColumn("คำตอบ", width="large")
            }
        )
        st.info(f"แสดง {len(filtered_df)} จาก {len(df_messages)} รายการ")
    else:
        st.info("ยังไม่มีข้อมูลข้อความ")

with tab2:
    st.markdown("### 📂 Chunks ที่ถูกดึงมาใช้งาน")
    if not df_chunks.empty:
        st.dataframe(
            df_chunks,
            use_container_width=True,
            height=400,
            column_config={
                "ID": st.column_config.NumberColumn("ID", width="small"),
                "Page Number": st.column_config.NumberColumn("หน้า", width="small"),
                "Source": st.column_config.TextColumn("แหล่งที่มา", width="medium"),
                "Chunk Text": st.column_config.TextColumn("เนื้อหา", width="large")
            }
        )
        st.info(f"จำนวน Chunks ทั้งหมด: {len(df_chunks)}")
    else:
        st.info("ยังไม่มีข้อมูล Chunks")

with tab3:
    st.markdown("### 📊 ข้อมูล Performance ของ LLM (50 รายการล่าสุด)")
    if not df_metrics.empty:
        st.dataframe(
            df_metrics,
            use_container_width=True,
            height=400,
            column_config={
                "ID": st.column_config.NumberColumn("ID", width="small"),
                "Timestamp": st.column_config.DatetimeColumn("เวลา", format="DD/MM/YYYY HH:mm"),
                "Prompt Tokens": st.column_config.NumberColumn("Prompt Tokens", width="small"),
                "Response Tokens": st.column_config.NumberColumn("Response Tokens", width="small"),
                "Response Time (s)": st.column_config.NumberColumn("Response Time", format="%.2f s", width="small")
            }
        )
    else:
        st.info("ยังไม่มีข้อมูล Metrics")

with tab4:
    st.markdown("### 💬 ความคิดเห็นจากผู้ใช้งาน")
    if not df_feedback.empty:
        # <<< CHANGED >>> ปรับตัวเลือกใน st.radio ให้สอดคล้องกับค่าใหม่
        filter_option = st.radio(
            "กรองตามประโยชน์ที่ได้รับ:", 
            ["ทั้งหมด", "ช่วยได้มาก 👍", "พอช่วยได้", "ยังไม่ช่วย 👎"], 
            horizontal=True
        )
        
        # <<< CHANGED >>> ปรับเงื่อนไขการ filter ให้ตรงกับตัวเลือกใหม่
        if filter_option == "ช่วยได้มาก 👍":
            display_feedback = df_feedback[df_feedback["Satisfaction"] == "ช่วยได้มาก 👍"]
        elif filter_option == "พอช่วยได้":
            display_feedback = df_feedback[df_feedback["Satisfaction"] == "พอช่วยได้"]
        elif filter_option == "ยังไม่ช่วย 👎":
            display_feedback = df_feedback[df_feedback["Satisfaction"] == "ยังไม่ช่วย 👎"]
        else: # "ทั้งหมด"
            display_feedback = df_feedback
        
        # ส่วนของ st.dataframe ยังคงเดิม ไม่ต้องแก้ไข
        st.dataframe(
            display_feedback,
            use_container_width=True,
            height=300,
            column_config={
                "ID": st.column_config.NumberColumn("ID", width="small"),
                "Timestamp": st.column_config.DatetimeColumn("เวลา", format="DD/MM/YYYY HH:mm"),
                "Satisfaction": st.column_config.TextColumn("ความพึงพอใจ", width="small"),
                "Feedback Text": st.column_config.TextColumn("ข้อเสนอแนะ", width="large")
            }
        )
        
        st.markdown("---")
        st.markdown("#### 🌟 Feedback ล่าสุด")

        # <<< CHANGED >>> สร้าง dictionary เพื่อ map ค่า satisfaction กับ emoji
        # วิธีนี้ทำให้โค้ดสะอาดและจัดการง่ายขึ้น
        emoji_map = {
            "ช่วยได้มาก 👍": "👍",
            "พอช่วยได้": "👌",
            "ยังไม่ช่วย 👎": "👎"
        }

        for idx, row in display_feedback.head(5).iterrows():
            # ใช้ .get() เพื่อดึง emoji จาก map, ถ้าไม่เจอก็ให้ใช้ค่า default "💬"
            emoji = emoji_map.get(row["Satisfaction"], "💬")
            
            with st.expander(f"{emoji} {row['User Message'][:60]}... - {row['Timestamp'].strftime('%d/%m/%Y %H:%M')}"):
                st.write(f"**ความพึงพอใจ:** {row['Satisfaction']}")
                if pd.notna(row['Feedback Text']) and row['Feedback Text']:
                    st.write(f"**ข้อเสนอแนะ:** {row['Feedback Text']}")
                else:
                    st.write("_ไม่มีข้อเสนอแนะเพิ่มเติม_")
    else:
        st.info("ยังไม่มีข้อมูลฟีดแบคจากผู้ใช้")

# ---------------------- Footer ----------------------
st.divider()
st.markdown("""
    <div style='text-align: center; color: #6b7280; padding: 1rem;'>
        <p>🔐 RAG Chatbot Admin Dashboard | Powered by Streamlit | Last Updated: {}</p>
    </div>
""".format(datetime.now().strftime('%H:%M:%S')), unsafe_allow_html=True)