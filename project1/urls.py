from django.urls import path
from . import views

app_name = 'project1'

urlpatterns = [
    # This triggers the index function in project1/views.py
    path('', views.index, name='index'), 
]