
// owl carousel //
$('.owl-carousel').owlCarousel({
    rtl: true,
    loop: true,
    margin: 220,
    nav: true,
    responsive: {
        0: {
            items: 1.42,
            margin: 130,
            nav: false,
        },
        600: {
            items: 3,
            margin: 420,
            nav: true,
        },
        1000:{
            items: 3,
            nav: true,
            margin: 120,
            mouseDrag: false,
        },
        1200:{
            items: 4,
            nav: true,
            margin: 270,
            mouseDrag: false,
        },
        1400: {
            items: 4,
            nav: true,
            margin: 100,
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
    rtl:true,
    spinner: true,
});
