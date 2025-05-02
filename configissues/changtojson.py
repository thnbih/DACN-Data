import os
import json
import xmltodict

# Đường dẫn tới folder chứa các file XML
input_folder = ".\hadoop"
output_folder = ".\hadoopjson"

# Tạo folder đầu ra nếu chưa có
os.makedirs(output_folder, exist_ok=True)

# Duyệt qua từng file trong folder
for filename in os.listdir(input_folder):
    if filename.endswith(".xml"):
        xml_path = os.path.join(input_folder, filename)

        with open(xml_path, 'r', encoding='utf-8') as xml_file:
            try:
                xml_data = xml_file.read()
                dict_data = xmltodict.parse(xml_data)
                json_data = json.dumps(dict_data, indent=4, ensure_ascii=False)

                json_filename = filename.replace(".xml", ".json")
                json_path = os.path.join(output_folder, json_filename)

                with open(json_path, 'w', encoding='utf-8') as json_file:
                    json_file.write(json_data)

                print(f"Đã chuyển {filename} -> {json_filename}")
            except Exception as e:
                print(f"Lỗi khi xử lý {filename}: {e}")
