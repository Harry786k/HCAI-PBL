from django.urls import path
from . import views

app_name = 'project2'

urlpatterns = [
    path('', views.index, name='index'),
    path('counterfactuals/', views.generate_counterfactuals, name='counterfactuals'), 
    path('feature_effects/', views.feature_effects, name='feature_effects'), # <-- NEW
]