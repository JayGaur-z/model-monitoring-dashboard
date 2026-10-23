from django.shortcuts import redirect

def home(request):
    """Clean redirect to the main active AutoML dashboard."""
    return redirect('dashboard')

