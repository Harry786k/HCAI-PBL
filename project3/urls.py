from django.urls import path
from . import views

urlpatterns = [
    path("expert/", views.expert_interface, name="expert_interface"),
]