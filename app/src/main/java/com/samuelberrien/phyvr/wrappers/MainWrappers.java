package com.samuelberrien.phyvr.wrappers;

import android.content.Context;
import android.content.res.AssetManager;
import com.samuelberrien.phyvr.utils.LoadImage;

public class MainWrappers {

	static {
		System.loadLibrary("phyvr");
	}

	private Context context;

	private long entitiesPtr;
	private long rendererPtr;
	private long enginePtr;
	private long playerPtr;
	private long controlPtr;

	private boolean vr;

	public MainWrappers(Context context, boolean vr) {
		this.context = context;
		this.vr = vr;
	}

	public void init() {
		LoadImage loadImage = new LoadImage(context, "heightmap/heightmap6.png");
		entitiesPtr = initEntity(context.getAssets(),
				loadImage.tofloatGreyArray(), loadImage.getWidth(), loadImage.getHeight());
		enginePtr = initEngine(entitiesPtr);
		rendererPtr = initRenderer(entitiesPtr);
		playerPtr = initPlayer(context.getAssets(), enginePtr, rendererPtr, entitiesPtr, vr);
		controlPtr = getControlPtrFromPlayer(playerPtr);
	}

	public long getControlPtr() {
		return controlPtr;
	}

	public void update() {
		updateEngine(enginePtr);
	}

	public void willDraw(float[] mHeadView, boolean VR) {
		willDrawRenderer(rendererPtr, mHeadView, VR);
	}

	public void draw(float[] mEyeProjectionMatrix,
					 float[] mEyeViewMatrix,
					 float[] myLighPosInEyeSpace,
					 float[] mCameraPos) {
		drawRenderer(rendererPtr,
				mEyeProjectionMatrix, mEyeViewMatrix, myLighPosInEyeSpace, mCameraPos);
	}

	public void free() {
		freeBoxes(entitiesPtr);
		freeLevel(enginePtr);
		freeRenderer(rendererPtr);
	}

	private native long initEntity(AssetManager assetManager, float[] heightmap, int width, int height);

	private native long initPlayer(AssetManager assetManager, long levelPtr, long rendererPtr, long entityPtr, boolean vr);

	private native long getControlPtrFromPlayer(long carPtr);

	private native long initEngine(long boxesPtr);

	private native long initRenderer(long boxesPtr);

	public native void addBox(AssetManager assetManager, long boxesPtr);

	private native void willDrawRenderer(long rendererPtr, float[] mHeadView, boolean VR);

	private native void drawRenderer(long rendererPtr,
									float[] mEyeProjectionMatrix,
									float[] mEyeViewMatrix,
									float[] myLighPosInEyeSpace,
									float[] mCameraPos);

	private native void updateEngine(long engineptr);

	private native void freeBoxes(long boxesPtr);

	private native void freeLevel(long levelPtr);

	private native void freeRenderer(long rendererPtr);
}
