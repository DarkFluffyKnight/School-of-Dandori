import streamlit as st
import pandas as pd
import ast

# Page Setup
st.set_page_config(page_title="School of Dandori | Course Portal", layout="wide")

# Visual Styling
st.markdown(
    """
    <style>
    .skill-tag {
        display: inline-block;
        background-color: #f0f4f8;
        color: #334e68;
        padding: 4px 12px;
        border-radius: 20px;
        margin: 2px;
        font-size: 0.85rem;
        border: 1px solid #bcccdc;
    }
    .price-badge {
        font-size: 1.4rem;
        font-weight: 700;
        color: #1b4332;
    }
    </style>
""",
    unsafe_allow_html=True,
)


# --- Helper Functions ---
def parse_list(data):
    try:
        return ast.literal_eval(data)
    except:
        return []


def get_all_unique_skills(df):
    all_skills = set()
    for skills_list in df["skills_developed"]:
        if isinstance(skills_list, list):
            all_skills.update(skills_list)
    return sorted(list(all_skills))


@st.cache_data
def load_and_clean_data():
    df = pd.read_csv("course_data.csv")
    df["cost"] = df["cost"].apply(lambda c: pd.to_numeric(c[1:]))
    for col in ["learning_objectives", "provided_materials", "skills_developed"]:
        df[col] = df[col].apply(parse_list)
    return df


# --- MAIN APP ---
try:
    df = load_and_clean_data()

    # --- SESSION STATE INITIALIZATION ---
    if "selected_skills" not in st.session_state:
        st.session_state.selected_skills = []

    # --- Sidebar ---
    st.sidebar.title("🌿 Dandori Menu")
    view_mode = st.sidebar.radio("View Mode:", ["Discovery Gallery", "Data Table View"])
    st.sidebar.divider()

    st.sidebar.subheader("Filter Your Search")
    search_query = st.sidebar.text_input("Search keywords:", "")

    slider_price = st.sidebar.slider(
        label="Price",
        min_value=df["cost"].min(),
        max_value=df["cost"].max(),
        value=(df["cost"].min(), df["cost"].max()),
    )

    selected_loc = st.sidebar.selectbox(
        "Location:",
        options=["All Locations"] + sorted(df["location"].unique().tolist()),
    )
    selected_type = st.sidebar.selectbox(
        "Course Category:",
        options=["All Categories"] + sorted(df["course_type"].unique().tolist()),
    )
    # ---------------------- SELECT INSTRUCTOR ---------------------
    selected_instructor = st.sidebar.selectbox(
        "Course Instructor:",
        options=["All Instructors"] + sorted(df["instructor"].unique().tolist()),
    )

    # --- UPDATED SKILLS MULTISELECT ---
    all_skills = get_all_unique_skills(df)
    selected_skills = st.sidebar.multiselect(
        "🎯 Skills Developed",
        options=all_skills,
        default=st.session_state.selected_skills,
        help="Select one or more skills to filter classes",
    )

    # Sync multiselect with session state
    if selected_skills != st.session_state.selected_skills:
        st.session_state.selected_skills = selected_skills
        st.rerun()  # Ensure the UI updates immediately with the state change

    # --- Filter Logic ---
    filtered_df = df.copy()
    if selected_loc != "All Locations":
        filtered_df = filtered_df[filtered_df["location"] == selected_loc]
    if selected_type != "All Categories":
        filtered_df = filtered_df[filtered_df["course_type"] == selected_type]
    # ---------------------- FILTER INSTRUCTOR ---------------------
    if selected_instructor != "All Instructors":
        filtered_df = filtered_df[filtered_df["instructor"] == selected_instructor]
    if st.session_state.selected_skills:
        filtered_df = filtered_df[
            filtered_df["skills_developed"].apply(
                lambda s: any(sk in s for sk in st.session_state.selected_skills)
            )
        ]
    if search_query:
        filtered_df = filtered_df[
            filtered_df["course_name"].str.contains(search_query, case=False)
            | filtered_df["course_description"].str.contains(search_query, case=False)
        ]

    filtered_df = filtered_df[
        (filtered_df["cost"] <= slider_price[1])
        & (filtered_df["cost"] >= slider_price[0])
    ]

    # --- Main Content Display ---
    if view_mode == "Discovery Gallery":
        st.title("School of Dandori")
        st.write(f"Showing **{len(filtered_df)}** whimsical classes.")
        st.divider()

        for _, row in filtered_df.iterrows():
            with st.container():
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.subheader(row["course_name"])
                    st.write(f"📍 **{row['location']}** | 👤 **{row['instructor']}**")

                    # Truncated description
                    desc = row["course_description"]
                    if len(desc) > 150:
                        st.write(f"{desc[:150]}...")
                        with st.expander("Read full description"):
                            st.write(desc)
                    else:
                        st.write(desc)

                    st.write("**Skills:**")
                    cols = st.columns(len(row["skills_developed"]))
                    for idx, skill in enumerate(row["skills_developed"]):
                        with cols[idx]:
                            # Check if skill is already selected
                            is_selected = skill in st.session_state.selected_skills
                            button_type = "primary" if is_selected else "secondary"

                            if st.button(
                                skill,
                                key=f"skill_{row['class_id']}_{idx}",
                                type=button_type,
                                use_container_width=True,
                            ):
                                # Toggle skill selection
                                if skill in st.session_state.selected_skills:
                                    st.session_state.selected_skills.remove(skill)
                                else:
                                    st.session_state.selected_skills.append(skill)
                                st.rerun()

                with col2:
                    st.markdown(
                        f'<p class="price-badge">{row["cost"]}</p>',
                        unsafe_allow_html=True,
                    )
                    with st.expander("Learning Objectives"):
                        for obj in row["learning_objectives"]:
                            st.write(f"• {obj}")
                    st.button("Book Now", key=f"btn_{row['class_id']}")
                st.divider()

    else:
        st.title("Admin Data View")
        # Apply the display formatting and headers you requested previously
        display_df = filtered_df[
            [
                "class_id",
                "course_name",
                "instructor",
                "location",
                "course_type",
                "cost",
                "skills_developed",
                "course_description",
            ]
        ]
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "class_id": "ID",
                "course_name": "Course Name",
                "instructor": "Instructor",
                "course_type": "Category",
                "location": "Location",
                "cost": st.column_config.NumberColumn("Cost", format="£ %.2f"),
                "skills_developed": "Skills",
                "course_description": "Full Description",
            },
        )
        st.download_button(
            "📥 Download Filtered CSV",
            data=filtered_df.to_csv(index=False),
            file_name="dandori_export.csv",
            mime="text/csv",
        )

except FileNotFoundError:
    st.error("Missing Data: Run your extraction script first!")
