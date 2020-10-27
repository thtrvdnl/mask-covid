from django.shortcuts import render


def index(request):
    if request.method == 'POST':
        print(request.Post.get('image_url', ''))
    context = {'data': 'hopa'}
    return render(request, 'mask/index.html', context)
