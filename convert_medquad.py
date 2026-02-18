import os
import pandas as pd
import xml.etree.ElementTree as ET

data = []

root_folder = "MedQuAD"   # change path if needed

for root, dirs, files in os.walk(root_folder):
    for file in files:
        if file.endswith(".xml"):
            file_path = os.path.join(root, file)
            tree = ET.parse(file_path)
            root_xml = tree.getroot()

            for qa in root_xml.findall(".//QAPair"):
                question = qa.findtext("Question")
                answer = qa.findtext("Answer")

                if question and answer:
                    data.append([question.strip(), answer.strip()])

df = pd.DataFrame(data, columns=["Question", "Answer"])
df.to_csv("medquad_qa.csv", index=False)

print("✅ Conversion complete! File saved as medquad_qa.csv")

