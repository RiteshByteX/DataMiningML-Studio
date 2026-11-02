from django.urls import path
from . import views

urlpatterns = [
    path('', views.upload_view, name='upload'),
    path('sample/', views.load_sample_view, name='load_sample'),
    path('eda/', views.eda_view, name='eda'),
    path('train/', views.train_view, name='train'),
    path('predict/', views.predict_view, name='predict'),
    path('download-processed/', views.download_processed_csv, name='download_processed'),
]
