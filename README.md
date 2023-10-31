# dlc-side-tracking-capsule

Port of the lim2 side-tracking pipline, using a trained deeplabcut model to ascribe annotated points to each frame in a new video. 

## model 

original model location: `/allen/aibs/technology/waynew/eye/np3_side_cam-sam_corbett-2020-03-31/dlc-models/iteration-0/np3_side_camMar31-trainset95shuffle1`

model data asset: [`np3_side_cam-sam_corbett-2020-03-31`](https://codeocean.allenneuraldynamics.org/data-assets/112d69a3-604d-4066-8171-8af8eaa14150/np3_side_cam-sam_corbett-2020-03-31)
- ID: `112d69a3-604d-4066-8171-8af8eaa14150`

[`side-tracking-test-video`](https://codeocean.allenneuraldynamics.org/data-assets/59ee4a48-3a12-416f-8aaa-895745f6622c/behavior-videos?filters=N4IgZglgNgLgpgJxALhAQylEAaEBnAewRhRAGME414ATHEGuPMlGBAVzlwAc0BzOCgCMPRAAV%2Bg5ACYADLjQ0AbmgB2ZOHWTAAvrlVoAtlPwRGAWjZoyAawiq%2BlpjHNKzcAvUbMEEbjAgCVVJpAAJDe3Z4ULx3ULdGAlCACihqRAx4iDgAd1CCMFCACyoabFCAIwIaAE9ywwI8GHzCmDRoAEpQsCJQ%2BCb7PlCaKDJQsjRuPHYoQVw2vjwUAG0GUfoKuCK0NyJ6fpJcBI96WIsrW0GQAF1cO1UtBmo0PDhD0wAvKSE5ABZpACcQgAbLhILMlsgRPgCOwEBoUKAiBA%2BPZSFACBMsHoQNxEBE8LEgpDQGh4UUIEopGxOLhGLN4KwOFwQJoICRkDSWXhtpQmbSQG5cvyWUKcgBlXnU5k4prpUiQAxYXAEHKqRAASUewNkAHYwDRpGAAeZZMCDeZfrqaGRzGg9QBWczSOByCpkaQVX4euYgVXqhAAOSMJgAQnBVKEABJkm0vGCzegUKi0ACCHJBAIAHLqhLIHQ7gQA6AEOgDMuYBoJAFRecAAqggsKgijAYFNkAB6TtkaoeDRqIsYWaqdVwjC1AwRMh4ItEPidmjPV4wPCdnX6w3G03mmiW622%2B26p0ut0er0%2BzsOgFwOC-NC-LPmMtoH6WkFgcxZtA-r%2Bl3W-A6YDAsC0jSCwuD2IwAAe4ptIyqC9oY3AMpo9BpE0GoPHA0FoVCwLZrm%2BagUWwIOrIshlg6vwKGQARUgA8mqiCQssoBmKQG4GkaJpmhaVo2najrOq6sjup63ouvQU5hhG0axhMTSJrgcCGO0zY1hGRa8nGSlwAAAsOEb2AMMBRHAc4IHwIA6LcID2NwUQACLPCsdnJukNDpgAMmgmwaQxdGhGW8ihHI0hlqEQi6sgsjSOh8ZYTBmi%2Bf5pCBc0IXlOFkXRbF8W4LEXypXAGlCEIRaPqEACyob0BAeAuW0K4ijoQA)
- ID: `59ee4a48-3a12-416f-8aaa-895745f6622c`

---

Developing a capsule in codoecean is a lot like developing in a local git repository: 
- you can clone from a remote (github) to get started
- changes are tracked as commits, with commit messages
- changes can be pushed or pulled from a remote

This template sets up a starting point for processing video data with `dlc`.

## for testing
get up and running quickly by *cloning this repo* in codeocean:
- open codeocean in a new tab [here](https://codeocean.allenneuraldynamics.org/)
- hit the `+` icon (top left) and select `"New Capsule" > "Clone from Git"` and paste the URL for this repo: `https://github.com/AllenNeuralDynamics/dlc-capsule-template`
- the capsule should open at this readme

## for more-permanent, collaborative capsule development
*create a new repo*, which can serve as the remote for one or more capsules:
- open this repository on github [here](https://github.com/AllenNeuralDynamics/dlc-capsule-template)
- hit the big green button to "`Use this template`": a new repo will be created after you decide its name
- follow the cloning instructions as per [`# for testing`](#for-testing), but supply the link to your new repo
- the capsule can now pull changes from github, so you can add or edit your files anywhere, push to github, then pull in codeocean
- to push changes *from* codeocean to github:
    - generate a personal access token for your account in github
    - add it to your account in codeocean
