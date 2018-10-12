from telegram.ext import Updater
import time
import threading
import socket

def get_ip():
    while True:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except:
            print("Oh no")
            time.sleep(5)

updater = Updater("644204064:AAE_zSKeWlwxhgcGp4vCdI7mZrtWYG1EipQ")

def shutdown():
    updater.stop()
    updater.is_idle = False

def send_message(bot, job):
    bot.send_message(chat_id="-317074198", text=get_ip())
    threading.Thread(target=shutdown).start()

def main():
    """Start the bot."""
    # # Start the Bot
    updater.start_polling()

    updater.job_queue.run_once(send_message, 0)

    # Run the bot until you press Ctrl-C or the process receives SIGINT,
    # SIGTERM or SIGABRT. This should be used most of the time, since
    # start_polling() is non-blocking and will stop the bot gracefully.
    updater.idle()

if __name__ == '__main__':
    main()
