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

	private long levelPtr;

	private boolean vr;

	private boolean isFree;
	private boolean isFreeable;

	private int levelIdx;

	public MainWrappers(Context context, boolean vr, int levelIdx) {
		this.context = context;
		this.vr = vr;
		isFree = false;
		isFreeable = false;
		this.levelIdx = levelIdx;
	}

	public void init() {
		levelPtr = makeLevel(levelIdx);
		enginePtr = makeEngine(levelPtr, context.getAssets());
		initLevel(context.getAssets(), vr, levelPtr, enginePtr);
		rendererPtr = makeRenderer(levelPtr);
		isFreeable = true;
	}

	public long getLevelPtr() {
		return levelPtr;
	}

	public void update() {
		updateEngine(enginePtr);
	}

	public boolean win() { return hasWon(levelPtr); }

	public boolean lose() { return hasLose(levelPtr); }

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

	public boolean isFree() {
		return isFree;
	}

	public void free() {
		if (isFreeable) {
			freeEngine(enginePtr);
			freeRenderer(rendererPtr);
			freeLevel(levelPtr);
			isFree = true;
			isFreeable = false;
		} else {
			throw new RuntimeException("Wrappers are already free");
		}
	}

	public boolean isInit() { return isFreeable; }

	private native long makeLevel(int id);
	private native void initLevel(AssetManager manager, boolean isVR, long levelPtr, long enginePtr);
	private native boolean hasWon(long levelPtr);
	private native boolean hasLose(long levelPtr);
	private native void freeLevel(long levelPtr);

	private native long makeEngine(long levelPtr, AssetManager manager);
	private native void updateEngine(long engineptr);
	private native void freeEngine(long levelPtr);

	private native long makeRenderer(long levelPtr);
	private native void willDrawRenderer(long rendererPtr, float[] mHeadView, boolean VR);
	private native void drawRenderer(long rendererPtr,
									 float[] mEyeProjectionMatrix,
									 float[] mEyeViewMatrix,
									 float[] myLighPosInEyeSpace,
									 float[] mCameraPos);
	private native void freeRenderer(long rendererPtr);
}
