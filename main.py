import asyncio
from bot_functions import dp, bot

# Starting the polling process for new updates
async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
 asyncio.run(main()) 