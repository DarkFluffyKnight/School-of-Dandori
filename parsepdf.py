import pdfplumber
import pandas as pd
import os
import re
import sys


def extract_boxed_skills(page):
    """
    Extracts words from a specific page area and groups them into skills
    based on their physical distance from one another.
    """
    words = page.extract_words()

    # 1. Find the boundaries for the skills section
    try:
        header = next(w for w in words if "Skills" in w["text"])
        footer = next(
            w
            for w in words
            if "Course" in w["text"] and "Description" in page.extract_text()
        )

        # Filter words between "Skills Developed" and "Course Description"
        skill_words = [w for w in words if header["bottom"] < w["top"] < footer["top"]]
    except:
        return []

    if not skill_words:
        return []

    # 2. Group words into distinct skill phrases
    skills = []
    if skill_words:
        current_skill = [skill_words[0]["text"]]
        for i in range(1, len(skill_words)):
            prev = skill_words[i - 1]
            curr = skill_words[i]

            # Distance Logic:
            # If the vertical distance is small AND the horizontal gap is small,
            # they belong to the same skill box.
            v_gap = abs(curr["top"] - prev["top"])
            h_gap = curr["x0"] - prev["x1"]

            # Threshold: < 5px vertical shift and < 10px horizontal gap
            if v_gap < 5 and h_gap < 10:
                current_skill.append(curr["text"])
            else:
                skills.append(" ".join(current_skill))
                current_skill = [curr["text"]]

        skills.append(" ".join(current_skill))

    return [s for s in skills if s.strip()]


def extract_course_data(path):

    with pdfplumber.open(path) as pdf:

        full_text = "\n".join([page.extract_text() for page in pdf.pages])
        # Split the text into a list of cleaned lines
        lines = [line.strip() for line in full_text.split("\n") if line.strip()]

        def find_line_after(keyword, lines):
            """Finds the line immediately following the one containing the keyword."""
            for i, line in enumerate(lines):
                if keyword in line:
                    return lines[i + 1] if i + 1 < len(lines) else None
            return None

        # Handle the two-column interleaved layout
        # Line 1: 'Instructor: Location:'
        # Line 2: 'Chef Waffleby Harrogate'
        instructor_location_line = find_line_after("Instructor:", lines)
        course_type_cost_line = find_line_after("Course Type:", lines)

        # Extracting and splitting values
        # We use rsplit or specific logic if names/locations have spaces
        instructor = None
        location = None
        if instructor_location_line:
            # Assuming the instructor name is the first part and location is the last
            # Adjust 'Harrogate' as the anchor if it's always the same location
            parts = instructor_location_line.split(" ")
            location = parts[-1]  # 'Harrogate'
            instructor = " ".join(parts[:-1])  # 'Chef Waffleby'
        course_type = None
        cost = None
        if course_type_cost_line:
            # 'Culinary Arts £75.00' -> ['Culinary', 'Arts', '£75.00']
            parts = course_type_cost_line.split(" ")
            cost = parts[-1]  # '£75.00'
            course_type = " ".join(parts[:-1])  # 'Culinary Arts'

        data = {
            "class_id": (
                re.search(r"CLASS_\d+", full_text).group(
                    0
                )  # pyright: ignore[reportOptionalMemberAccess]
                if re.search(r"CLASS_\d+", full_text)
                else None
            ),  #
            "course_name": lines[0],  # 'The Art of Wondrous Waffle Weaving'
            "instructor": instructor,
            "course_type": course_type,
            "location": location,
            "cost": cost,
            # Objectives and Materials (Regex)
            "learning_objectives": [
                re.sub(r"^•\s*", "", l)
                for l in lines
                if "•" in l and lines.index(l) < lines.index("Provided Materials")
            ],  # Learning objectivs and provided materials are the only
            # things with bullet points and separated by "Provided Materials"
            "provided_materials": [
                re.sub(r"^•\s*", "", l)
                for l in lines
                if "•" in l and lines.index(l) > lines.index("Provided Materials")
            ],
            "skills_developed": (
                extract_boxed_skills(pdf.pages[1]) if len(pdf.pages) > 1 else []
            ),  # Split by double space
            "course_description": " ".join(
                lines[
                    lines.index("Course Description")
                    + 1 : lines.index(next(l for l in lines if "Class ID" in l))
                ]
            ),
        }

        return data


def process_pdf_folder(folder_path):
    """Iterates through a folder and returns a consolidated DataFrame."""
    all_data = []

    # List all PDF files in the directory, filter by name
    files = [
        f
        for f in os.listdir(folder_path)
        if f.endswith(".pdf") and f.startswith("class_")
    ]
    # Load data from each one
    for filename in files:
        path = os.path.join(folder_path, filename)
        print(f"Processing: {filename}...")
        try:
            row = extract_course_data(path)
            all_data.append(row)
        except Exception as e:
            print(f"Error in {filename}: {e}")
            break

    return pd.DataFrame(all_data)


def main(path="./course_pdfs"):
    df = process_pdf_folder(path)
    print("Saving to course_data.csv")
    df.to_csv("course_data.csv", index=False)


if __name__ == "__main__":
    try:
        main(sys.argv[1])
    except:
        main()
