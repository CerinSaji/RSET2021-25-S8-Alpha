document.addEventListener("DOMContentLoaded", () => {
    const splashScreen = document.querySelector(".splash-screen");
    const loginContainer = document.querySelector(".login-container");

    // Show login form after splash animation ends
    splashScreen.addEventListener("animationend", () => {
        splashScreen.style.display = "none";
        loginContainer.style.display = "block";
    });

    // Handle login form submission
    document.querySelector(".login-form").addEventListener("submit", async function(event) {
        event.preventDefault();

        const email = document.getElementById("email").value;
        const password = document.getElementById("password").value;

        try {
            const response = await fetch("http://localhost:5000/login", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ email, password })
            });

            const data = await response.json();
            if (response.ok) {
                // Store email in localStorage to use in main.html
                localStorage.setItem("userEmail", email);

                // Redirect to main.html
                window.location.href = "main.html";
            } else {
                alert(data.message);
            }
        } catch (error) {
            console.error("Error:", error);
            alert("Something went wrong. Please try again.");
        }
    });
});
