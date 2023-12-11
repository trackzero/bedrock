# Started with examples from https://github.com/awsdocs/aws-doc-sdk-examples/blob/main/python/example_code/
#   Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#   SPDX-License-Identifier: Apache-2.0
#
# customized by github.com/trackzero

"""
Purpose

Shows how to use the AWS SDK for Python (Boto3) with the Amazon Bedrock Runtime client
to run inferences using Bedrock models.
"""

import asyncio
import base64
import json
import logging
import os
import random

import boto3
from botocore.exceptions import ClientError

my_region = "us-west-2"
my_model = "stable-diffusion-xl"  # image gen model to use.
logger = logging.getLogger(__name__)

class BedrockRuntimeWrapper:
    """Encapsulates Amazon Bedrock Runtime actions."""

    def __init__(self, bedrock_runtime_client):
        """
        :param bedrock_runtime_client: A low-level client representing Amazon Bedrock Runtime.
                                       Describes the API operations for running inference using
                                       Bedrock models.
        """
        self.bedrock_runtime_client = bedrock_runtime_client

    def invoke_stable_diffusion(self, prompt, seed, style_preset=None):
        """
        Invokes the Stability.ai Stable Diffusion XL model to create an image using
        the input provided in the request body.

        :param prompt: The prompt that you want Stable Diffusion  to use for image generation.
        :param seed: Random noise seed (omit this option or use 0 for a random seed)
        :param style_preset: Pass in a style preset to guide the image model towards
                             a particular style.
        :return: Base64-encoded inference response from the model.
        """

        try:
            # The different model providers have individual request and response formats.
            # For the format, ranges, and available style_presets of Stable Diffusion models refer to:
            # https://platform.stability.ai/docs/api-reference#tag/v1generation

            body = {
                "text_prompts": [{"text": prompt}],
                "seed": seed,
                "cfg_scale": 10,
                "steps": 30,
            }

            if style_preset:
                body["style_preset"] = style_preset

            response = self.bedrock_runtime_client.invoke_model(
                modelId="stability.stable-diffusion-xl", body=json.dumps(body)
            )

            response_body = json.loads(response["body"].read())
            base64_image_data = response_body["artifacts"][0]["base64"]

            return base64_image_data

        except ClientError:
            logger.error("Couldn't invoke Stable Diffusion XL")
            raise

    def invoke_titan_image(self, prompt, seed):
        """
        Invokes the Titan Image model to create an image using the input provided in the request body.

        :param prompt: The prompt that you want Amazon Titan to use for image generation.
        :param seed: Random noise seed (range: 0 to 2147483647)
        :return: Base64-encoded inference response from the model.
        """

        try:
            # The different model providers have individual request and response formats.
            # For the format, ranges, and default values for Titan Image models refer to:
            # https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-titan-image.html

            request = json.dumps(
                {
                    "taskType": "TEXT_IMAGE",
                    "textToImageParams": {"text": prompt},
                    "imageGenerationConfig": {
                        "numberOfImages": 4,
                        "quality": "standard",
                        "cfgScale": 8.0,
                        "height": 1024,
                        "width": 1024,
                        "seed": seed,
                    },
                }
            )

            response = self.bedrock_runtime_client.invoke_model(
                modelId="amazon.titan-image-generator-v1", body=request
            )

            response_body = json.loads(response["body"].read())
            base64_image_data = response_body["images"][0]

            return base64_image_data

        except ClientError:
            logger.error("Couldn't invoke Titan Image model")
            raise


def save_image(base64_image_data, model):
    output_dir = "output/" + model

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    i = 1
    while os.path.exists(os.path.join(output_dir, f"image_{i}.png")):
        i += 1

    image_data = base64.b64decode(base64_image_data)

    file_path = os.path.join(output_dir, f"image_{i}.png")
    with open(file_path, "wb") as file:
        file.write(image_data)

    return file_path


def invoke(wrapper, model_id, prompt, style_preset=None):
    print("-" * 88)
    print(f"Invoking: {model_id}")
    print("Prompt: " + prompt)

    try:
        if model_id == "stability.stable-diffusion-xl":
            seed = random.randint(0, 4294967295)
            base64_image_data = wrapper.invoke_stable_diffusion(
                prompt, seed, style_preset
            )
            image_path = save_image(base64_image_data, "diffusion")
            print(f"The generated image has been saved to {image_path}")

        elif model_id == "amazon.titan-image-generator-v1":
            seed = random.randint(0, 2147483647)
            base64_image_data = wrapper.invoke_titan_image(prompt, seed)
            image_path = save_image(base64_image_data, "titan")
            print(f"The generated image has been saved to {image_path}")

    except ClientError:
        logger.exception("Couldn't invoke model %s", model_id)
        raise


def usage_demo():
    """
    Demonstrates the invocation of various large-language and image generation models:
    Anthropic Claude 2, AI21 Labs Jurassic-2, and Stability.ai Stable Diffusion XL.
    """
    logging.basicConfig(level=logging.INFO)
    print("-" * 88)
    print("Welcome to the Amazon Bedrock Runtime demo.")
    print("-" * 88)

    client = boto3.client(service_name="bedrock-runtime", region_name=my_region)

    wrapper = BedrockRuntimeWrapper(client)

    image_generation_prompt = input("Enter your image prompt") #no error handling, as is my way.

    image_style_preset = "photographic"

    invoke(
        wrapper,
        "stability.stable-diffusion-xl",
        image_generation_prompt,
        image_style_preset,
    )

    invoke(wrapper, "amazon.titan-image-generator-v1", image_generation_prompt)


if __name__ == "__main__":
    usage_demo()

# snippet-end:[python.example_code.bedrock-runtime.BedrockRuntimeWrapper.class]