import os
from openai import OpenAI
import time

# === CẤU HÌNH API ===
GROK_API_KEY = "xai-F4YW5B77elAiMK11BWXRqe4VvGZ820pwkd6q0VDnAMJ8wtAlykMsCsgLtGjGxzK0c90BOiJDuLtTgqzO"  # ← Nhớ thay bằng API key thật

client = OpenAI(
    api_key=GROK_API_KEY,
    base_url="https://api.x.ai/v1",
)

# === GỌI API GROK ĐỂ TẠO FILE CẤU HÌNH ===
def generate_config_file(issue_description, file_type='mysql'):
    prompt_template = """You are a configuration file generator. Based on the following issue description, generate a {file_type} config file with the specified benign. Ensure it reflects the benign clearly and is syntactically correct. Do not explain, just output the config content.
Ensure that request results in a fully developed configuration file that is both realistic and complete, containing all necessary settings and parameters to fully represent the issue described. The generated file should not be minimal or overly simplified. Every generated configuration file must be usable in a real-world scenario.

Description: {issue_description}
"""
    prompt = prompt_template.format(file_type=file_type, issue_description=issue_description)


    response = client.chat.completions.create(
        model="grok-3-mini",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=4096
    )

    # Lấy nội dung trả về
    return response.choices[0].message.content.strip()

# === HÀM CHÍNH ĐỂ ĐỌC FILE MÔ TẢ VÀ LƯU CẤU HÌNH ===
def generate_all_configs(description_file_path, output_dir, file_type='mysql'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(description_file_path, 'r', encoding='utf-8') as f:
        descriptions = [line.strip() for line in f if line.strip()]

    for idx, desc in enumerate(descriptions, start=1):
        print(f"[+] Generating config {idx} ...")
        try:
            content = generate_config_file(desc, file_type=file_type)
            ext = 'cnf' if file_type == 'mysql' else 'conf'
            output_path = os.path.join(output_dir, f"3config_{idx}.{ext}")
            with open(output_path, 'w', encoding='utf-8') as out_file:
                out_file.write(content)
            print(f"    -> Saved to {output_path}")
        except Exception as e:
            print(f"    [!] Error at config {idx}: {e}")
        time.sleep(1)  # Giới hạn 9 request/giây

    print(f"\n✅ Đã sinh {len(descriptions)} file cấu hình lỗi vào thư mục `{output_dir}`.")

# === CHẠY ===
if __name__ == "__main__":
    generate_all_configs(
        description_file_path="config_descriptions.txt",
        output_dir="output_configs_benign",
        file_type="mysql"  # có thể thay bằng nginx, docker, apache, systemd, ...
    )
