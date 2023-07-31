<!DOCTYPE html>
<html>
<head>
  <title>Welcome to My Website</title>
  <style>
    body {
      font-family: Arial, sans-serif;
      margin: 0;
      padding: 0;
    }

    #welcomeMessage {
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      height: 100vh;
      text-align: center;
    }

    #welcomeMessage h1 {
      font-size: 48px;
      color: #333;
      margin-bottom: 20px;
      opacity: 0;
    }

    #welcomeMessage p {
      font-size: 24px;
      color: #666;
      opacity: 0;
    }
  </style>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/gsap/3.9.1/gsap.min.js"></script>
  <script>
    document.addEventListener("DOMContentLoaded", function() {
      gsap.fromTo(
        "#welcomeMessage",
        { opacity: 0, y: -100 },
        {
          opacity: 1,
          y: 0,
          duration: 1.5,
          ease: "power4.out",
          stagger: 0.2,
          onComplete: function() {
            gsap.to("#welcomeMessage h1, #welcomeMessage p", {
              opacity: 1,
              y: 0,
              duration: 1.5,
              ease: "power4.out",
            });
          },
        }
      );
    });
  </script>
</head>
<body>
  <div id="welcomeMessage">
    <h1>Welcome to My Website!</h1>
    <p>We're glad you're here.</p>
  </div>
</body>
</html>
