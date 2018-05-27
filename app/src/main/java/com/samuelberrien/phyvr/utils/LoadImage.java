package com.samuelberrien.phyvr.utils;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;

import java.io.IOException;
import java.io.InputStream;

public class LoadImage {

	private boolean imageLoaded;
	private Bitmap bitmap;

	public LoadImage(Context context, String imgAssetPath) {
		try {
			InputStream in = context.getAssets().open(imgAssetPath);
			BitmapFactory.Options options = new BitmapFactory.Options();
			options.inSampleSize = 8;
			bitmap = BitmapFactory.decodeStream(in, null, options);
			imageLoaded = true;
		} catch (IOException ioe) {
			imageLoaded = false;
			ioe.printStackTrace();
		}
	}

	private float[] tofloatRGBArray() {
		if (!imageLoaded) return new float[0];

		int width = getWidth();
		int height = getHeight();

		float[] res = new float[width * height * 3];
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				int color = bitmap.getPixel(j, i);
				res[(i * width + j) * 3] = Color.red(color);
				res[(i * width + j) * 3 + 1] = Color.green(color);
				res[(i * width + j) * 3 + 2] = Color.blue(color);
			}
		}
		return res;
	}

	public int getWidth() {
		return bitmap.getWidth();
	}

	public int getHeight() {
		return bitmap.getHeight();
	}

	public float[] tofloatGreyArray() {
		float[] rgbArray = tofloatRGBArray();
		float[] res = new float[rgbArray.length / 3];
		for (int i = 0; i < res.length; i++) {
			int j = i * 3;
			res[i] = (rgbArray[j] + rgbArray[j + 1] + rgbArray[j + 2]) / 3.f;
			res[i] /= 255.f;
		}
		return res;
	}
}
