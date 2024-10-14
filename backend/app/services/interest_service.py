from core.db import get_db_connection

async def save_user_interests(user_id, interests):
    conn = await get_db_connection()

    sql = """
    INSERT INTO UserInterests (user_id, interest)
    VALUES (%s, %s)
    ON DUPLICATE KEY UPDATE count = count + 1;
    """
    
    try:
        async with conn.cursor() as cursor:
            for interest in interests:
                await cursor.execute(sql, (user_id, interest))
            await conn.commit()
    finally:
        conn.close()

async def get_user_interest(user_id):
    conn = await get_db_connection()

    try:
        async with conn.cursor() as cursor:
            await cursor.execute(
                """
                SELECT COUNT(*) FROM UserInterests
                WHERE user_id = %s AND selected = FALSE
                """, (user_id,)
            )
            result = await cursor.fetchone()
            all_selected = result[0] == 0

            if all_selected:
                await cursor.execute(
                    """
                    UPDATE UserInterests
                    SET selected = FALSE
                    WHERE user_id = %s
                    """, (user_id,)
                )
                await conn.commit()

            await cursor.execute(
                """
                SELECT interest_id, interest FROM UserInterests
                WHERE user_id = %s AND selected = FALSE
                ORDER BY count DESC, created_at ASC
                LIMIT 1
                """, (user_id,)
            )
            interest = await cursor.fetchone()
            return {"interest_id": interest[0], "interest": interest[1]}
    finally:
        conn.close()


async def mark_interest(interest_id):
    conn = await get_db_connection()

    try:
        async with conn.cursor() as cursor:
            await cursor.execute(
                """
                UPDATE UserInterests
                SET selected = TRUE
                WHERE interest_id= %s
                """, (interest_id,)
            )
            await conn.commit()
    finally:
        conn.close()