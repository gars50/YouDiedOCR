from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from process_thread import ProcessThread
from forms import InputVideoForm
import os

app = Flask(__name__)

#We don't need to preserve sessions between restarts of the application
app.secret_key = os.urandom(24).hex()

process_threads = []

@app.route("/", methods=['POST','GET'])
def main():
    form = InputVideoForm()
    if form.validate_on_submit():
        global process_threads
        process_thread = ProcessThread(form.youtube_url, form.game, form.debug, form.milliseconds_skipped, form.seconds_skipped_after_death)
        process_thread.start()
        process_threads.append(process_thread)
        thread_id = len(process_threads) - 1
        return redirect(url_for('process', thread_id=thread_id))
    elif request.method == 'GET':
        return render_template('index.html', form=form)
    else:
        flash('Error processing request.', 'error')

@app.route("/process/<int:thread_id>", methods=['GET'])
def process(thread_id):
    return render_template('process.html', thread_id=thread_id)

@app.route("/check_progress/<int:thread_id>", methods=['GET'])
def check_progress(thread_id):
    global process_threads
    return jsonify(process_threads[thread_id].progress_status)