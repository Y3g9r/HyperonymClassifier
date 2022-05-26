from flask import render_template, redirect
from app import app
from HyperonymClassifier import Hyperonym_handler as hh
from .forms import LoginForm

NERUAL_HANDLER = hh.hyperonym_handler()

@app.route('/index', methods = ['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        NERUAL_HANDLER.sentence_parse(form.openid.data)
        return redirect('/index')
    return render_template('index.html',
        title = 'Hyperonyms page',
        form = form)