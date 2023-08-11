
// owl carousel //
$('.owl-carousel').owlCarousel({
    rtl: true,
    loop: true,
    margin: 10,
    nav: true,
    responsive: {
        0: {
            items: 1.10,
            margin: 10,
            nav: false,
        },
        600: {
            items: 3,
            nav: true,
        },
        1000: {
            items: 3,
            nav: true,
            mouseDrag: false,
        }
    }
})

var owlss = document.getElementsByClassName('owl-nav');
for (var i = 0; i < owlss.length; i++) {
    owlss[i].className = "owl-nav";
    owlss[i].onclick = function showNav() {
        for (var j = 0; j < owlss.length; j++) {
            owlss[j].className = "owl-nav";
        }
    }
}

var lightbox = new SimpleLightbox('.gallery a', {
    overlay: true,
    nav: true,
    spinner: true,
    navText: ["<img src={% static 'ReastApi/images/prev.png' %}/>", "<img src={% static 'ReastApi/images/next.png' %} />"],
});
