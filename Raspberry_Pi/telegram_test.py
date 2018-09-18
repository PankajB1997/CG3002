from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
import re
import time
import threading
from urllib import request, error

def get_ip():
    while True:
        try:
            req = request.urlopen("http://checkip.dyndns.org").read()
            return re.findall(b"\d{1,3}.\d{1,3}.\d{1,3}.\d{1,3}", req)[0].decode('utf-8')
        except:
            print("Oh no")
            time.sleep(5)

updater = Updater("644204064:AAE_zSKeWlwxhgcGp4vCdI7mZrtWYG1EipQ")

def shutdown():
    updater.stop()
    updater.is_idle = False

def send_message(bot, job):
    bot.send_message(chat_id="620733342", text=get_ip())
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
