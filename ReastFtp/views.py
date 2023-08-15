from django.views.generic import TemplateView

class ShowHome(TemplateView):
    template_name = "ReastFtp/home/index.html"

class ShowSignIn(TemplateView):
    template_name = "ReastFtp/auth/signIn.html"

class ShowSignUp(TemplateView):
    template_name = "ReastFtp/auth/signUp.html"
