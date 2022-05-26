from flask import render_template, flash, redirect
from app import app
from .forms import LoginForm

@app.route('/index', methods = ['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        #flash('Login requested for OpenID="' + form.openid.data + '", remember_me=' + str(form.remember_me.data))
        return redirect('/index')
    return render_template('index.html',
        title = 'Hyperonyms page',
        form = form)