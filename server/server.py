import numpy as np
import argparse
import logging
import asyncio
import json
import uuid
import ssl
import cv2
import os

from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaBlackhole, MediaPlayer, MediaRecorder

# from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
# from tensorflow.keras.preprocessing.image import img_to_array
# from tensorflow.keras.models import load_model

from dataclasses import dataclass

from aiohttp import web
from av import VideoFrame

import time

ROOT = os.path.dirname(__file__)

logger = logging.getLogger("pc")
pcs = set()


@dataclass
class Path:
    prototxtPath: str = "utils/deploy.prototxt"
    weightsPath: str = "utils/res10_300x300_ssd_iter_140000.caffemodel"
    mobile: str = "utils/mobile.h5"


@dataclass
class Predict:
    mask = None


class VideoTransformTrack(MediaStreamTrack):
    """
    A video stream track that transforms frames from an another track.
    """

    kind = "video"

    def __init__(self, track, transform):
        super().__init__()
        self.track = track
        self.transform = transform

        # self.faceNet = cv2.dnn.readNet(Path.prototxtPath, Path.weightsPath)
        # self.maskNet = load_model(Path.mobile)
        self.frame_counter = 0

    async def recv(self):
        frame = await self.track.recv()
        self.frame_counter += 1

        # if self.transform == "mask":
        #     if (self.frame_counter % 5) == 0:
        #         img = frame.to_ndarray(format="bgr24")
        #         (h, w) = img.shape[:2]
        #         blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), (104.0, 177.0, 123.0))
        #         self.faceNet.setInput(blob)
        #         detections = self.faceNet.forward()
        #         faces = []
        #         locs = []
        #         preds = []
        #         for i in range(0, detections.shape[2]):
        #             confidence = detections[0, 0, i, 2]
        #             if confidence > 0.5:
        #                 box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        #                 (startX, startY, endX, endY) = box.astype("int")
        #                 (startX, startY) = (max(0, startX), max(0, startY))
        #                 (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
        #
        #                 face = img[startY:endY, startX:endX]
        #                 try:
        #                     face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        #                     face = cv2.resize(face, (224, 224))
        #                     face = img_to_array(face)
        #                     face = preprocess_input(face)
        #                 except Exception as exp:
        #                     print(f'Error: {exp}')
        #                     return frame
        #
        #                 faces.append(face)
        #                 locs.append((startX, startY, endX, endY))
        #
        #         if len(faces) > 0:
        #             faces = np.array(faces, dtype="float32")
        #             preds = self.maskNet.predict(faces, batch_size=32)
        #             print('{:.2f}, {:.2f}'.format(preds[0][0], preds[0][1]))
        #             Predict.mask = preds[0][0]
        #
        #         # rebuild a VideoFrame, preserving timing information
        #         new_frame = VideoFrame.from_ndarray(img, format="bgr24")
        #         new_frame.pts = frame.pts
        #         new_frame.time_base = frame.time_base
        #         return new_frame
        #     else:
        #         return frame

        # else:
        return frame


async def index(request):
    content = open(os.path.join(ROOT, "index.html"), "r").read()
    return web.Response(content_type="text/html", text=content)


async def javascript(request):
    content = open(os.path.join(ROOT, "client.js"), "r").read()
    return web.Response(content_type="application/javascript", text=content)


async def offer(request):
    try:
        params = await request.json()
        offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

        pc = RTCPeerConnection()
        pc_id = "PeerConnection(%s)" % uuid.uuid4()
        pcs.add(pc)

        def log_info(msg, *args):
            logger.info(pc_id + " " + msg, *args)

        log_info("Created for %s", request.remote)

        # prepare local media
        player = MediaPlayer(os.path.join(ROOT, "utils/demo-instruct.wav"))

        recorder = MediaBlackhole()

        @pc.on("datachannel")
        def on_datachannel(channel):
            @channel.on("message")
            def on_message(message):
                # if isinstance(message, str) and message.startswith("mask") and Predict.mask is not None:
                #     channel.send('Predict {:.2f}'.format(Predict.mask))
                if isinstance(message, str) and message.startswith("mask"):
                    channel.send('Predict Mask 100')

        @pc.on("iceconnectionstatechange")
        async def on_iceconnectionstatechange():
            log_info("ICE connection state is %s", pc.iceConnectionState)
            if pc.iceConnectionState == "failed":
                await pc.close()
                pcs.discard(pc)

        @pc.on("track")
        def on_track(track):
            log_info("Track %s received", track.kind)

            if track.kind == "audio":
                pc.addTrack(player.audio)
                recorder.addTrack(track)
            elif track.kind == "video":
                local_video = VideoTransformTrack(
                    track, transform=params["video_transform"]
                )
                pc.addTrack(local_video)

            @track.on("ended")
            async def on_ended():
                log_info("Track %s ended", track.kind)
                await recorder.stop()

        # handle offer
        await pc.setRemoteDescription(offer)
        await recorder.start()

        # send answer
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
            ),
        )
    except Exception as exp:
        print(exp)

async def on_shutdown(app):
    # close peer connections
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()


async def server_web():
    app = web.Application()

    app.on_shutdown.append(on_shutdown)
    app.add_routes([web.get('/', index)])
    app.add_routes([web.get("/client.js", javascript)])
    app.add_routes([web.post('/offer', offer)])

    # web.run_app(
    #     app, access_log=None, host=args.host, port=args.port, ssl_context=ssl_context
    # )
    return app

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="WebRTC audio / video / data-channels demo"
    )
    parser.add_argument("--cert-file", help="SSL certificate file (for HTTPS)")
    parser.add_argument("--key-file", help="SSL key file (for HTTPS)")
    parser.add_argument(
        "--host", default="0.0.0.0", help="Host for HTTP server (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=8080, help="Port for HTTP server (default: 8080)"
    )
    parser.add_argument("--verbose", "-v", action="count")
    parser.add_argument("--write-audio", help="Write received audio to a file")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    if args.cert_file:
        ssl_context = ssl.SSLContext()
        ssl_context.load_cert_chain(args.cert_file, args.key_file)
    else:
        ssl_context = None

    app = web.Application()
    app.on_shutdown.append(on_shutdown)
    app.router.add_get("/", index)
    app.router.add_get("/client.js", javascript)
    app.router.add_post("/offer", offer)
    web.run_app(
        app, access_log=None, host=args.host, port=args.port, ssl_context=ssl_context
    )
