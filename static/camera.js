async function startCamera() {
    const video = document.getElementById("video");

    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: true
        });

        video.srcObject = stream;
    } catch (err) {
        console.error("Camera error:", err);
        alert("Camera access denied or not working");
    }
}

