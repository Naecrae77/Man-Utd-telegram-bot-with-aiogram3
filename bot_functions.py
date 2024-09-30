from typing_extensions import Text
import logging
from openai import OpenAI
import os
from dotenv import load_dotenv
from aiogram import Bot, Dispatcher, types
from aiogram.filters.command import Command
from df import df
from query_proc_functions import ask

# Enable logging so you don't miss important messages
logging.basicConfig(level=logging.INFO)

load_dotenv()
keys = os.getenv('api_token')

# Bot object
bot = Bot(token=keys)
# Dispatcher
dp = Dispatcher()

db_entries = df.shape[0]

# Handler for the /start command
@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    url = 'https://i.imghippo.com/files/BfnzT1727361497.jpg'
    await message.bot.send_photo(
    chat_id=message.chat.id,
    photo=url,
    caption=f"Hello {(message.from_user.full_name)}, welcome to the Man United history bot. Here you can know anything you wish to know about the greatest football club on the planet. \n\nFeel free to ask any questions related to our club. The more detailed your question, the more detailed response you get. \n\nFor any inquiries, you can contact the developer @Naecrae \n\nDon't know what to do? type /help",
    )

#help handler
@dp.message(Command('help'))
async def cmd_help(message: types.Message):
    await message.answer(f'This theme of this bot is Manchester United history. \n\nPress /start to restart or initialise the bot. \n/help shows the "help" menu. \n\nThis bot has been trained using data from articles about Manchester United from around the web. There are currently {db_entries} entries in the database \n\nFeel free to ask the bot anything about Man United e.g "When did man utd first win the premier league?"')

#handler for sending and recieving chatgpt messages
@dp.message()
async def gpt(message: types.Message):
    user_id = message.from_user.id
    user_query = message.text

    # Send typing action and temporary message
    await bot.send_chat_action(message.chat.id, action="typing")
    temp_message = await message.reply("Please wait while the bot fetches a reply...")

    # Get response from ChatGPT (assuming ask is defined)
    gpt_response = ask(user_query)

    # Edit the temporary message with the actual response
    await temp_message.edit_text(gpt_response)
