# bot.py

import asyncio
import os
import io
import contextlib
import importlib
from aiogram import Bot, Dispatcher, F, types
from aiogram.filters import CommandStart
from aiogram.types import FSInputFile, InlineKeyboardMarkup, InlineKeyboardButton
from aiogram.fsm.state import StatesGroup, State
from aiogram.fsm.context import FSMContext
from aiogram.fsm.storage.memory import MemoryStorage

# ===== Настройки =====
BOT_TOKEN = "8364995071:AAGcP6peDccXjyKcJH9bcM-IiXVqc67RFeE"
GRAPH_FOLDER = "graphs"
os.makedirs(GRAPH_FOLDER, exist_ok=True)

# Импорт модулей
import MaximiN
import min2

# ===== Машина состояний =====
class GraphStates(StatesGroup):
    choose_action = State()
    choose_mode = State()
    choose_file = State()

# ===== Инициализация =====
bot = Bot(BOT_TOKEN)
dp = Dispatcher(storage=MemoryStorage())

# ===== /start =====
@dp.message(CommandStart())
async def cmd_start(message: types.Message, state: FSMContext):
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="1️⃣ Посчитать готовую нумерацию", callback_data="action_count")],
            [InlineKeyboardButton(text="2️⃣ Найти оптимальную", callback_data="action_optimize")]
        ]
    )
    await message.answer("👋 Привет!\nЯ бот для анализа графов.\n\nВыбери действие:", reply_markup=keyboard)
    await state.set_state(GraphStates.choose_action)

# ===== Выбор действия =====
@dp.callback_query(GraphStates.choose_action)
async def choose_action_callback(callback: types.CallbackQuery, state: FSMContext):
    if callback.data == "action_count":
        await state.update_data(action="count")
        await send_file_list(callback.message, state)
        await state.set_state(GraphStates.choose_file)
    else:
        await state.update_data(action="optimize")
        keyboard = InlineKeyboardMarkup(
            inline_keyboard=[
                [InlineKeyboardButton(text="1️⃣ Maximin / Minimax", callback_data="mode_maximin")],
                [InlineKeyboardButton(text="2️⃣ Min / Max", callback_data="mode_min")]
            ]
        )
        await callback.message.answer("⚙️ Выбери режим оптимизации:", reply_markup=keyboard)
        await state.set_state(GraphStates.choose_mode)

# ===== Выбор режима (для оптимизации) =====
@dp.callback_query(GraphStates.choose_mode)
async def choose_mode_callback(callback: types.CallbackQuery, state: FSMContext):
    if callback.data == "mode_maximin":
        await state.update_data(mode="maximin")
    else:
        await state.update_data(mode="min")

    await send_file_list(callback.message, state)
    await state.set_state(GraphStates.choose_file)

# ===== Выбор файла =====
async def send_file_list(message: types.Message, state: FSMContext):
    files = [f for f in os.listdir(GRAPH_FOLDER) if f.endswith(".graph")]
    inline_buttons = [[InlineKeyboardButton(text=f, callback_data=f"file_{f}")] for f in files]
    inline_buttons.append([InlineKeyboardButton(text="📎 Загрузить новый файл", callback_data="upload_new")])
    keyboard = InlineKeyboardMarkup(inline_keyboard=inline_buttons)
    await message.answer("📂 Выбери .graph файл:", reply_markup=keyboard)

@dp.callback_query(GraphStates.choose_file)
async def choose_file_callback(callback: types.CallbackQuery, state: FSMContext):
    if callback.data == "upload_new":
        await callback.message.answer("Отправь мне .graph файл 📎")
        return
    else:
        filename = callback.data.replace("file_", "")
        file_path = os.path.join(GRAPH_FOLDER, filename)
        await process_graph(callback.message, state, file_path)

@dp.message(GraphStates.choose_file, F.document)
async def handle_file(message: types.Message, state: FSMContext):
    file = message.document
    file_path = os.path.join(GRAPH_FOLDER, file.file_name)
    await bot.download(file, destination=file_path)
    await process_graph(message, state, file_path)

# ===== Основная обработка =====
async def process_graph(message: types.Message, state: FSMContext, file_path: str):
    data = await state.get_data()
    action = data.get("action")
    mode = data.get("mode")  # None, если выбран подсчёт готовой

    await message.answer("✅ Файл получен, начинаю обработку...")

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        try:
            if action == "count":
                importlib.reload(min2)
                min2.main(file_path, choice="1")
            else:
                if mode == "maximin":
                    importlib.reload(MaximiN)
                    MaximiN.main(file_path, choice="2")
                else:
                    importlib.reload(min2)
                    min2.main(file_path, choice="2")
        except Exception as e:
            await message.answer(f"❌ Ошибка при выполнении: {e}")
            return

    output_text = buffer.getvalue()
    if len(output_text) > 4000:
        out_file = "output.txt"
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(output_text)
        await message.answer_document(FSInputFile(out_file), caption="📄 Результаты расчёта")
    else:
        await message.answer(f"📊 Результаты:\n\n{output_text[:4000]}")

    # ===== Выбор изображений в зависимости от режима =====
    if action == "count":
        images = ["graph_from_file.png"]
    elif mode == "maximin":
        images = ["graph_maximin.png", "graph_minimax.png"]
    else:
        images = ["graph_min.png", "graph_max.png"]

    # Отправляем только существующие изображения
    for img in images:
        if os.path.exists(img):
            await message.answer_photo(FSInputFile(img))

    await message.answer("✅ Готово!")
    await state.clear()
    await cmd_start(message, state)

# ===== Обработка лишних сообщений =====
@dp.message()
async def fallback(message: types.Message):
    await message.answer("Используй /start, чтобы начать работу.")

# ===== Запуск =====
if __name__ == "__main__":
    print("✅ Бот запущен. Нажми Ctrl+C для выхода.")
    asyncio.run(dp.start_polling(bot))
