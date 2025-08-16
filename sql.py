import streamlit as st
import pandas as pd
import requests
import pyodbc
import sqlite3
import oracledb
import os
import io
import tempfile
import plotly.express as px
import random
from faker import Faker
from PIL import Image
import numpy as np
import base64

st.set_page_config(page_title="SQL Chatbot with Visualization", layout="wide")

# --- Helper function: Convert image to base64 ---
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        return base64.b64encode(f.read()).decode()

# --- File paths ---
logo_path = "logo.png"
bg_img_path = "poiuygtfdsx.PNG"

# --- Base64 encode ---
logo_base64 = get_base64_of_bin_file(logo_path)
bg_img_base64 = get_base64_of_bin_file(bg_img_path)

# --- CSS Styling ---
st.markdown(f"""
<style>
    /* White background for main content */
    .stApp {{
        background-color: white !important;
    }}

    /* Hide default Streamlit header */
    header[data-testid="stHeader"] {{
        display: none !important;
    }}

    /* Sidebar background styling */
    section[data-testid="stSidebar"] {{
        background-color: rgba(0,0,0,0.85);
        background-image: url("data:image/png;base64,{bg_img_base64}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        padding-top: 20px !important;
        color: white;
    }}

    /* Sidebar text styling */
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] div {{
        color: #ffffff !important;
        font-size: 1.05rem;
        font-weight: 600;
    }}

    /* Main titles */
    h1, h2, h3 {{
        color: #004d66;
        text-shadow: 1px 1px #b2ebf2;
    }}

    /* Cards and charts */
    .stDataFrame, .stPlotlyChart {{
        background-color: rgba(255,255,255,0.85);
        border-radius: 12px;
        padding: 10px;
        box-shadow: 0px 4px 8px rgba(0,0,0,0.1);
    }}

    /* Buttons */
    div.stButton>button {{
        background: linear-gradient(45deg, #42a5f5, #7e57c2);
        color: white;
        border-radius: 8px;
        padding: 8px 16px;
        font-weight: bold;
        border: none;
    }}

    /* Inputs */
    input, textarea, select {{
        border-radius: 6px;
        border: 1px solid #ccc;
    }}

    /* Sidebar logo styling */
    .sidebar-logo {{
        display: flex;
        justify-content: center;
        margin-top: -35px; /* Upar move kiya */
        margin-bottom: 15px;
    }}
    .sidebar-logo img {{
        max-height: 40px;
        width: auto;
    }}
</style>
""", unsafe_allow_html=True)

# Sidebar logo placement
st.sidebar.markdown(
    f"""
    <div class="sidebar-logo">
        <img src="data:image/png;base64,{logo_base64}">
    </div>
    """,
    unsafe_allow_html=True
)



import oracledb


# Excel Download
def add_excel_download_button(df, key):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name="Sheet1")
    st.download_button("📥 Download Excel", data=output.getvalue(), file_name="query_results.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key=key)

# SQL Generator
def generate_sql(user_input, db_type, available_tables):
    API_KEY = st.secrets["groq"]["api_key"]   
    url = "https://api.groq.com/openai/v1/chat/completions"
    prompt = f"You are an assistant that writes SQL queries for {'Oracle' if db_type == 'Oracle' else 'SQL Server'} databases. Available tables and columns: {', '.join(available_tables)}. ONLY return SQL query."
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    data = {"model": "llama3-8b-8192", "messages": [{"role": "system", "content": prompt}, {"role": "user", "content": user_input}], "temperature": 0.2}
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content'].strip().replace("```", "")
    return f"❌ Error {response.status_code}: {response.text}"

st.title("💬 SQL Chatbot with Visualization")

with st.sidebar:
    
    st.header("⚙️ Configuration")
    mode = st.radio("Choose Mode", ["Natural → SQL", "Upload & Query", "Temporary DB", "Connect to DB"])
    db_type = st.radio("Database Type", ["SQL Server", "Oracle"])

    

st.session_state.setdefault("conn", None)
st.session_state.setdefault("dataframes", {})
st.session_state.setdefault("temp_db_path", None)

# Mode 1: Natural → SQL

if mode == "Natural → SQL":
    st.subheader("💬 Describe your query in English")
    user_input = st.text_input("🗣️ Ask your question", placeholder="e.g., Show all customers from Karachi")

    if st.button("⚡ Generate SQL"):
        if not user_input.strip():
            st.warning("⚠️ Please enter a valid query.")
        else:
            with st.spinner("🔍 Generating SQL..."):
                API_KEY = st.secrets["groq"]["api_key"]
                url = "https://api.groq.com/openai/v1/chat/completions"

                prompt = f"You are an assistant that writes SQL queries for {'Oracle' if db_type == 'Oracle' else 'SQL Server'} databases. ONLY return the SQL query."

                headers = {
                    "Authorization": f"Bearer {API_KEY}",
                    "Content-Type": "application/json"
                }

                data = {
                    "model": "llama3-8b-8192",
                    "messages": [
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": user_input}
                    ],
                    "temperature": 0.2
                }

                response = requests.post(url, headers=headers, json=data)

                if response.status_code == 200:
                    sql = response.json()['choices'][0]['message']['content'].strip().replace("```", "")
                    st.success("✅ SQL Generated:")
                    st.code(sql, language="sql")
                else:
                    st.error(f"❌ Error {response.status_code}: {response.text}")
# mode 2
elif mode == "Upload & Query":
    import re  # for fixing SQL syntax

    file = st.file_uploader("📂 Upload CSV or Excel", type=["csv", "xlsx"])

    if file:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            st.session_state.temp_db_path = tmp.name
        conn = sqlite3.connect(st.session_state.temp_db_path, check_same_thread=False)
        st.session_state.conn = conn
        st.session_state.dataframes = {}

        if file.name.endswith(".csv"):
            df = pd.read_csv(file)
            df.to_sql("data_table", conn, index=False, if_exists="replace")
            st.session_state.dataframes = {"data_table": df}
        else:
            xls = pd.ExcelFile(file)
            for sheet in xls.sheet_names:
                df = xls.parse(sheet)
                table_name = sheet.replace(" ", "_")
                df.to_sql(table_name, conn, index=False, if_exists="replace")
                st.session_state.dataframes[table_name] = df

        st.success("✅ File uploaded and tables created.")

    if st.session_state.get("dataframes"):
        selected_table = st.selectbox("📁 Select Table", list(st.session_state.dataframes.keys()))
        df = st.session_state.dataframes[selected_table]
        conn = st.session_state.conn

        # ✅ Show full table
        st.markdown("### 📋 Table Preview")
        st.dataframe(df)
        # add_excel_download_button(df, key="excel_mode2")

        # ✅ Visualization
        if st.checkbox("📈 Enable Visualization"):
            st.markdown("### 📊 Create Custom Graph")
            x_col = st.selectbox("X-axis", df.columns, key="x1")
            y_col = st.selectbox("Y-axis", df.columns, key="y1")
            chart = st.selectbox("Chart Type", ["Bar", "Line", "Area", "Pie"], key="chart1")
            color = st.color_picker("Pick a color", "#f54291", key="color1")

            df_plot = df[[x_col, y_col]].dropna()
            try:
                if chart == "Bar":
                    fig = px.bar(df_plot, x=x_col, y=y_col, color_discrete_sequence=[color])
                elif chart == "Line":
                    fig = px.line(df_plot, x=x_col, y=y_col, color_discrete_sequence=[color])
                elif chart == "Area":
                    fig = px.area(df_plot, x=x_col, y=y_col, color_discrete_sequence=[color])
                elif chart == "Pie":
                    if not pd.api.types.is_numeric_dtype(df_plot[y_col]):
                        st.warning("⚠️ Pie chart 'values' must be numeric.")
                        fig = None
                    else:
                        fig = px.pie(df_plot, names=x_col, values=y_col)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"❌ Chart Error: {e}")
        # ✅ Insert Excel Data into Database (ONLY in Mode 2)
        st.markdown("---")
        st.subheader("🗄️ Insert Excel Data into Database")
        enable_insert = st.checkbox("Enable Database Insert")

        if enable_insert:
            if db_type == "SQL Server":
                server = st.text_input("Server", "localhost\\SQLEXPRESS")
                database = st.text_input("Database", "BankDb")
                username = st.text_input("Username (leave blank for Windows Auth)", value="")
                password = st.text_input("Password", type="password")
                table_name = st.text_input("Table Name")

                if st.button("Insert into SQL Server"):
                    try:
                        conn_str = f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={server};DATABASE={database};"
                        conn_str += f"UID={username};PWD={password};" if username.strip() else "Trusted_Connection=yes;"
                        conn = pyodbc.connect(conn_str)
                        cursor = conn.cursor()

                        # 🔹 Auto-generate insert query
                        cols = ",".join(df.columns)
                        placeholders = ",".join(["?"] * len(df.columns))
                        insert_query = f"INSERT INTO {table_name} ({cols}) VALUES ({placeholders})"

                        for row in df.itertuples(index=False, name=None):
                            cursor.execute(insert_query, row)

                        conn.commit()
                        st.success("✅ Data inserted into SQL Server successfully!")
                    except Exception as e:
                        st.error(f"❌ Error: {e}")

            elif db_type == "Oracle":
                dsn = st.text_input("Oracle DSN", "hostname:1521/servicename")
                user = st.text_input("Username")
                password = st.text_input("Password", type="password")
                table_name = st.text_input("Table Name")

                if st.button("Insert into Oracle"):
                    try:
                        conn = oracledb.connect(user=user, password=password, dsn=dsn)
                        cursor = conn.cursor()

                        cols = ",".join(df.columns)
                        placeholders = ",".join([":" + str(i+1) for i in range(len(df.columns))])
                        insert_query = f"INSERT INTO {table_name} ({cols}) VALUES ({placeholders})"

                        cursor.executemany(insert_query, df.values.tolist())
                        conn.commit()
                        st.success("✅ Data inserted into Oracle successfully!")
                    except Exception as e:
                        st.error(f"❌ Error: {e}")



        # ✅ Natural Language Query
        st.markdown("---")
        st.markdown("### 💬 Ask in Natural Language (Query the Table)")
        nl_query = st.text_input("🗣️ Example: Show total profit per category with positive profit")

        def fix_sql_for_sqlite(sql):
            sql = sql.strip()

            # Convert TOP N to LIMIT
            match = re.search(r"SELECT\s+TOP\s+(\d+)", sql, re.IGNORECASE)
            if match:
                limit_val = match.group(1)
                sql = re.sub(r"SELECT\s+TOP\s+\d+", "SELECT", sql, flags=re.IGNORECASE)
                if "LIMIT" not in sql.upper():
                    sql += f" LIMIT {limit_val}"

            # Basic replacements
            sql = sql.replace("GETDATE()", "datetime('now')")
            sql = sql.replace("ISNULL(", "IFNULL(")
            sql = sql.replace("NVARCHAR", "TEXT").replace("VARCHAR", "TEXT")
            sql = sql.replace("IDENTITY(1,1)", "AUTOINCREMENT")
            sql = re.sub(r"\bGO\b", "", sql, flags=re.IGNORECASE)
            sql = re.sub(r"\bUSE\s+\w+;", "", sql, flags=re.IGNORECASE)

            # Convert WHERE SUM(...) to HAVING
            where_sum_pattern = r"WHERE\s+(SUM\([^)]+\)\s*[<>=!]+\s*[\d.]+)"
            if re.search(where_sum_pattern, sql, re.IGNORECASE):
                condition = re.search(where_sum_pattern, sql, re.IGNORECASE).group(1)
                sql = re.sub(where_sum_pattern, "", sql, flags=re.IGNORECASE)
                if "HAVING" in sql.upper():
                    sql = re.sub(r"(HAVING\s+)", r"\1" + condition + " AND ", sql, flags=re.IGNORECASE)
                else:
                    sql += f" HAVING {condition}"

            if "ORDER BY" in sql.upper() and "LIMIT" not in sql.upper():
                sql += " LIMIT 1"

            return sql

        if st.button("⚡ Generate Query"):
            sql = generate_sql(
                nl_query + " (Note: Use SQLite syntax. Do not use TOP N or SQL Server-specific keywords.)",
                db_type="SQLite",
                available_tables=[selected_table]
            )
            fixed_sql = fix_sql_for_sqlite(sql)
            st.session_state.generated_sql_mode2 = fixed_sql

        if st.session_state.get("generated_sql_mode2"):
            edited_sql = st.text_area("✏️ Edit SQL before running", st.session_state.generated_sql_mode2, height=150)

            if st.button("▶️ Run Query"):
                try:
                    if edited_sql.strip().lower().startswith("select"):
                        df_result = pd.read_sql(edited_sql, conn)
                        st.success("✅ Query Result:")
                        st.dataframe(df_result)
                        add_excel_download_button(df_result, key="excel_mode2_query")
                    else:
                        cursor = conn.cursor()
                        for statement in edited_sql.strip().split(";"):
                            if statement.strip():
                                cursor.execute(statement)
                        conn.commit()
                        st.success("✅ Non-SELECT query executed.")
                        st.rerun()
                except Exception as e:
                    st.error(f"❌ SQL Execution Error: {e}")
        
#mode 3
elif mode == "Temporary DB":
    st.subheader("📝 Create & Manage Temporary Tables")

    if not st.session_state.get("temp_db_path"):
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            st.session_state.temp_db_path = tmp.name

    conn = sqlite3.connect(st.session_state.temp_db_path, check_same_thread=False)
    st.session_state.conn = conn

    # 🧱 Table Creation
    table_name = st.text_input("Enter Table Name")
    num_cols = st.number_input("Number of Columns", min_value=1, max_value=10, value=3)

    col_defs, col_names, col_names_for_pk, col_types = [], [], [], []

    for i in range(num_cols):
        cname = st.text_input(f"Column {i+1} Name", key=f"cname_{i}")
        ctype = st.selectbox(f"Column {i+1} Type", ["TEXT", "INTEGER", "REAL"], key=f"ctype_{i}")
        if cname:
            col_names_for_pk.append(cname)
            col_names.append(cname)
            col_types.append(ctype)

    auto_inc_col = st.selectbox("🔢 Select Auto-Increment Column (Optional, must be INTEGER)", ["None"] + col_names_for_pk)

    for cname, ctype in zip(col_names, col_types):
        if cname == auto_inc_col and ctype == "INTEGER":
            col_defs.append(f"{cname} INTEGER PRIMARY KEY AUTOINCREMENT")
        else:
            col_defs.append(f"{cname} {ctype}")

    if st.button("Create Table"):
        if table_name and col_defs:
            create_query = f"CREATE TABLE IF NOT EXISTS {table_name} ({', '.join(col_defs)})"
            conn.execute(create_query)
            conn.commit()
            st.success(f"✅ Table '{table_name}' created.")
            st.rerun()

    # 🔍 Show Tables
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]
    selected_table = st.selectbox("📂 Select Table", tables)

    if selected_table:
        df = pd.read_sql(f"SELECT * FROM {selected_table}", conn)

        # ✅ Success messages
        if "update_success" in st.session_state:
            st.success("✅ Row updated successfully.")
            del st.session_state.update_success
        if "delete_success" in st.session_state:
            st.success("✅ Row deleted successfully.")
            del st.session_state.delete_success

        st.dataframe(df, use_container_width=True)
        add_excel_download_button(df, key="excel_temp")

        # 🤖 AI SQL Generator with editable field
        st.markdown("### 🤖 Ask AI to Generate SQL")
        nl_query = st.text_input("🗣️ Enter command like 'Insert 5 rows in the table'")

        if st.button("⚡ Generate SQL"):
            sql = generate_sql(nl_query, db_type="SQLite", available_tables=[selected_table])
            if selected_table.lower() not in sql.lower():
                st.warning(f"⚠️ Ensure prompt includes '{selected_table}' to affect the right table.")
            st.session_state.generated_sql = sql

        if "generated_sql" in st.session_state:
            editable_sql = st.text_area("📝 Edit the generated SQL if needed", value=st.session_state.generated_sql, height=200)

            if st.button("▶️ Run SQL"):
                try:
                    if editable_sql.strip().lower().startswith("select"):
                        df_result = pd.read_sql(editable_sql, conn)
                        st.dataframe(df_result)
                    else:
                        for stmt in editable_sql.strip().split(";"):
                            if stmt.strip():
                                cursor.execute(stmt)
                        conn.commit()
                        st.success("✅ SQL executed successfully.")
                        st.rerun()
                except Exception as e:
                    st.error(f"❌ SQL Execution Error: {e}")

        # ➕ Manual Insert
        st.markdown("### ➕ Insert Row (Manual Only)")
        manual_values = []
        hide_cols = []

        if auto_inc_col != "None":
            hide_cols.append(auto_inc_col)

        for col in df.columns:
            if col in hide_cols:
                st.info(f"(Auto Increment Column '{col}' is skipped in manual entry)")
                manual_values.append(None)
                continue
            val = st.text_input(f"Enter {col}", key=f"manual_{col}")
            manual_values.append(val)

        if st.button("Insert"):
            try:
                insert_cols = [col for col in df.columns if col not in hide_cols]
                insert_values = [val for col, val in zip(df.columns, manual_values) if col not in hide_cols]
                placeholders = ",".join(["?"] * len(insert_values))
                col_clause = ",".join(insert_cols)
                conn.execute(f"INSERT INTO {selected_table} ({col_clause}) VALUES ({placeholders})", insert_values)
                conn.commit()
                st.success("✅ Row inserted.")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Insert Error: {e}")

# ✅ Mode 4: Connect to DB
elif mode == "Connect to DB":
    st.subheader("🔌 Database Connection")

    server = st.text_input("Server", value="localhost\\SQLEXPRESS" if db_type == "SQL Server" else "172.16.1.230")
    database = st.text_input("Database", value="BankDb" if db_type == "SQL Server" else "pdborcl.scilife.net")
    username = st.text_input("Username (leave blank for Windows Auth)", value="", key="user")
    password = st.text_input("Password", type="password", key="pwd")

    if st.button("🔗 Connect"):
        try:
            if db_type == "SQL Server":
                conn_str = f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={server};DATABASE={database};"
                conn_str += f"UID={username};PWD={password};" if username.strip() else "Trusted_Connection=yes;"
                conn = pyodbc.connect(conn_str)
                cursor = conn.cursor()
                cursor.execute("SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE = 'BASE TABLE'")
                tables = [row[0] for row in cursor.fetchall()]
            else:
                dsn = oracledb.makedsn(server, 1521, service_name=database)
                conn = oracledb.connect(user=username, password=password, dsn=dsn)
                cursor = conn.cursor()
                cursor.execute("SELECT table_name FROM user_tables")
                tables = [row[0] for row in cursor.fetchall()]

            st.session_state.conn = conn
            st.session_state.tables = tables
            st.success("✅ Connected to database.")
        except Exception as e:
            st.error(f"❌ Connection Error: {e}")

    # ✅ If Connected
    if st.session_state.get("conn") and st.session_state.get("tables"):
        selected_table = st.selectbox("📁 Select table to view", st.session_state.tables)

        if selected_table:
            try:
                df_table = pd.read_sql(f"SELECT * FROM {selected_table}", st.session_state.conn)
                st.markdown("### 📋 Table Preview")
                st.dataframe(df_table)
                add_excel_download_button(df_table, key="excel_mode4_table")

                # ✅ Visualization (only if checkbox is ticked)
                if st.checkbox("📈 Enable Visualization"):
                    st.markdown("### 📊 Create Custom Graph")
                    x_col = st.selectbox("X-axis", df_table.columns, key="x2")
                    y_col = st.selectbox("Y-axis", df_table.columns, key="y2")
                    chart = st.selectbox("Chart Type", ["Bar", "Line", "Area", "Pie"], key="chart2")
                    color = st.color_picker("Pick a color", "#f54291", key="color2")

                    df_plot = df_table[[x_col, y_col]].dropna()

                    try:
                        if chart == "Bar":
                            fig = px.bar(df_plot, x=x_col, y=y_col, color_discrete_sequence=[color])
                        elif chart == "Line":
                            fig = px.line(df_plot, x=x_col, y=y_col, color_discrete_sequence=[color])
                        elif chart == "Area":
                            fig = px.area(df_plot, x=x_col, y=y_col, color_discrete_sequence=[color])
                        elif chart == "Pie":
                            fig = px.pie(df_plot, names=x_col, values=y_col)

                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"❌ Chart Error: {e}")
            except Exception as e:
                st.error(f"❌ Table Load Error: {e}")

        # ✅ Natural Language to SQL
        st.markdown("---")
        st.markdown("### 💬 Ask in Natural Language (Query the Table)")
        nl_query = st.text_input("🗣️ Example: Add GPA column and update for 4 students")

        if st.button("⚡ Generate SQL"):
            sql = generate_sql(nl_query, db_type=db_type, available_tables=st.session_state.tables)
            st.session_state.generated_sql_mode4 = sql

        if "generated_sql_mode4" in st.session_state:
            editable_sql = st.text_area("📝 Review or Edit Generated SQL", value=st.session_state.generated_sql_mode4, height=200)

            if st.button("▶️ Run SQL"):
                try:
                    if editable_sql.strip().lower().startswith("select"):
                        df_result = pd.read_sql(editable_sql, st.session_state.conn)
                        st.success("✅ Query Result:")
                        st.dataframe(df_result)
                        add_excel_download_button(df_result, key="excel_mode4_query")
                    else:
                        cursor = st.session_state.conn.cursor()
                        for stmt in editable_sql.strip().split(";"):
                            if stmt.strip():
                                cursor.execute(stmt)
                        st.session_state.conn.commit()
                        st.success("✅ Query executed successfully.")
                        st.rerun()
                except Exception as e:
                    st.error(f"❌ SQL Execution Error: {e}")

