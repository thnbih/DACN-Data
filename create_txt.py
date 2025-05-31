import random

# Danh sách mẫu cho các phần cấu hình
sections = ['[mysqld]', '[client]', '[mysql]', '[mysqldump]', '[mysqld_safe]']
settings = {
    'max_connections': ['100', '200', '500', '1000'],
    'innodb_buffer_pool_size': ['128M', '512M', '1G', '2G'],
    'sql_mode': ['STRICT_ALL_TABLES', 'NO_ENGINE_SUBSTITUTION', 'TRADITIONAL'],
    'port': ['3306', '3307', '3308'],
    'bind-address': ['127.0.0.1', '0.0.0.0'],
    'character-set-server': ['utf8mb4', 'latin1'],
    'default-storage-engine': ['InnoDB', 'MyISAM'],
    'log-error': ['/var/log/mysql/error.log', '/tmp/mysql_error.log'],
    'slow_query_log': ['1', '0'],
    'slow_query_log_file': ['/var/log/mysql/slow.log', '/tmp/slow.log']
}

# Hàm tạo mô tả cấu hình
def generate_description(index):
    section = random.choice(sections)
    setting = random.sample(list(settings.keys()), k=3)
    description = f"such as section with sectting {section} and settings "
    description += ", ".join([f"{s}={random.choice(settings[s])}" for s in setting])
    description += " to optimize MySQL performance and reliability."
    if index % 2 == 0:
        description += " This configuration is suitable for high-traffic applications."
    else:
        description += " This configuration is ideal for development and testing environments."
    return description

# Ghi ra file
with open("mysql_cnf_descriptions.txt", "w", encoding="utf-8") as f:
    for i in range(400):
        desc = generate_description(i)
        f.write(desc + "\n")

print("Đã tạo xong file mysql_cnf_bengin_descriptions.txt với 400 mô tả.")
