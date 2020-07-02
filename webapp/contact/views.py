from django.shortcuts import render
import datetime
from .models import ContactForm


def contact_form(request):

    now = datetime.datetime.now()
    year = now.year
    month = now.month
    day = now.day

    if len(str(day)) == 1:
        day = "0" + str(day)
    if len(str(month)) == 1:
        month = "0" + str(month)

    today = str(year) + "/" + str(month) + "/" + str(day)
    time = str(now.hour) + "/" + str(now.minute)

    if request.method == 'POST':

        fname = request.POST.get('fname')
        lname = request.POST.get('lname')
        email = request.POST.get('email')
        subject = request.POST.get('subject')
        message = request.POST.get('message')

        b = ContactForm(fname=fname,lname=lname,email=email,subject=subject,date=today,time=time,message=message)
        b.save()

        return render(request, 'front/msgbox.html')

    return render(request,'front/msgbox.html')
