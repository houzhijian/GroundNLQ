##https://github.com/srama2512/NaQ/blob/main/utils/download_features.py

import argparse
import os
import os.path as osp
import subprocess as sp

naq_root = "/s1_md0/leiji/v-zhijian/NaQ"


DOWNLOAD_URLS = {
    "clip": "https://utexas.box.com/shared/static/vp0n2h6imytpqhiqrwlh0y1uveqe0ten.zip",
    "slowfast": [
        "https://utexas.box.com/shared/static/phlgzezvc4onn6zalbpeg6pr8hbuphw7.zip",
        "https://utexas.box.com/shared/static/4o3hy4hpp6di08nlc8oqfft0ga2u0l1t.z01",
        "https://utexas.box.com/shared/static/8hrfbsl714p591iwkrk02z06rfgwf6ce.z02",
        "https://utexas.box.com/shared/static/hsoii0p3wj3hpe8ht0403ju3a5u0s8qh.z03",
        "https://utexas.box.com/shared/static/d9mbzrfkeydhzasmg71h1ghdqr6jacvq.z04",
        "https://utexas.box.com/shared/static/cvvepuoifnaaia7iq6k9b2gawr40awr8.z05",
        "https://utexas.box.com/shared/static/hvzshqdlso3y2o4lzguhk5jlrc5at8kf.z06",
        "https://utexas.box.com/shared/static/cf2uv2r8ctpyyyir3fs2dypwcepjgyiv.z07",
        "https://utexas.box.com/shared/static/lhz1nhm1n3rv3bmo65g5twhz2pwxsjib.z08",
        "https://utexas.box.com/shared/static/34gyrnykqea3jkfeobfkhmq41iylh0yb.z09",
        "https://utexas.box.com/shared/static/4l2kz39t7yq3wfqhuuwk0lr7arcso1fu.z10",
    ],
    "egovlp": "https://utexas.box.com/shared/static/ixinsqrbscbnbpmeg1ac5ebyz16rz9km.zip",
    "internvideo": [
        "https://utexas.box.com/shared/static/aikunogcjyy678fh3tb7n0ct2fq5ackk.zip",
        "https://utexas.box.com/shared/static/hlpf68z8xdf24qt64wy9d0pfcvjyf7eg.z01",
        "https://utexas.box.com/shared/static/e34ibqqhuq6ygy1xbo8o8r95yk2ubj0v.z02",
    ],
}


def download_features(feature_type):
    assert (
        feature_type in DOWNLOAD_URLS
    ), f"{feature_type} not in {DOWNLOAD_URLS.keys()}"
    os.makedirs(osp.join(naq_root, "data/features"), exist_ok=True)
    print(f"===> Downloading {feature_type} features")
    urls = DOWNLOAD_URLS[feature_type]
    if type(urls) == str:
        destination = osp.join(naq_root, f"data/features/{feature_type}.zip")
        sp.call(["wget", "-O", destination, DOWNLOAD_URLS[feature_type]])
    else:
        # Download split zip files
        for idx, url in enumerate(urls):
            if idx == 0:
                destination = osp.join(
                    naq_root, f"data/features/{feature_type}_split.zip"
                )
            else:
                destination = osp.join(
                    naq_root, f"data/features/{feature_type}_split.z{idx:02d}"
                )
            sp.call(["wget", "-O", destination, url])
        # Merge zip files into one
        print("Merging split zip files. Might take a while...")
        sp.call(
            [
                "zip",
                "-q",
                "-s",
                "0",
                osp.join(naq_root, f"data/features/{feature_type}_split.zip"),
                "--out",
                osp.join(naq_root, f"data/features/{feature_type}.zip"),
            ]
        )
        # Delete split zip files
        for idx in range(len(urls)):
            if idx == 0:
                destination = osp.join(
                    naq_root, f"data/features/{feature_type}_split.zip"
                )
            else:
                destination = osp.join(
                    naq_root, f"data/features/{feature_type}_split.z{idx:02d}"
                )
            sp.call(["rm", destination])

    print(f"Extracting features...")
    sp.call(
        [
            "unzip",
            "-qq",
            "-d",
            osp.join(naq_root, "data/features"),
            osp.join(naq_root, f"data/features/{feature_type}.zip"),
        ]
    )

    print(f"Cleaning up...")
    # Move features to destination directory
    if feature_type == "clip":
        sp.call(
            [
                "mv",
                osp.join(naq_root, "data/features/Slowfast_vitb16CLIP_videofeature"),
                osp.join(naq_root, "data/features/clip"),
            ]
        )
    elif feature_type == "egovlp":
        sp.call(
            [
                "mv",
                osp.join(naq_root, "data/features/all_egovlp_feats"),
                osp.join(naq_root, "data/features/egovlp"),
            ]
        )
    elif feature_type == "internvideo":
        sp.call(
            [
                "mv",
                osp.join(naq_root, "data/features/all-videointern-prefusion"),
                osp.join(naq_root, "data/features/internvideo"),
            ]
        )
    elif feature_type == "slowfast":
        sp.call(
            [
                "mv",
                osp.join(naq_root, "data/features/official"),
                osp.join(naq_root, "data/features/slowfast"),
            ]
        )
    else:
        raise ValueError(f"Unknown feature type: {feature_type}")

    # Delete zip file
    sp.call(["rm", osp.join(naq_root, f"data/features/{feature_type}.zip")])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--feature_types",
        nargs="+",
        type=str,
        default="egovlp",
        choices=list(DOWNLOAD_URLS.keys()),
    )
    args = parser.parse_args()

    for feature_type in args.feature_types:
        download_features(feature_type)
