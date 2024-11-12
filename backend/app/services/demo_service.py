from core.db import get_db_connection
import re
import os

def clean_string(text: str) -> str:
    return re.sub(r'[^가-힣0-9\s\-:\[\]]', '', text)


def parse_memory_txt_file(file_path: str):
    memories_dict = {}
    if not os.path.exists(file_path):
        return memories_dict

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            cleaned_line = clean_string(line)
            match = re.match(r'(\d{4}-\d{2}-\d{2})\s*[:\-]?\s*(.+)', cleaned_line)
            if match:
                date, content = match.groups()
                print(date, content)
                if date not in memories_dict:
                    memories_dict[date] = []
                memories_dict[date].append(content)
    memories = [{"date": date, "diary": diary_entries} for date, diary_entries in memories_dict.items()]
    return memories


async def get_user_interests(user_id: str):
    conn = await get_db_connection()

    try:
        async with conn.cursor() as cursor:
            await cursor.execute(
                """
                SELECT interest, count FROM UserInterests
                WHERE user_id = %s
                ORDER BY count DESC
                """, (user_id,)
            )
            interests = await cursor.fetchall()
            return [{"keyword": interest, "count": str(count)} for interest, count in interests]
    finally:
        conn.close()