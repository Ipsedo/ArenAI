package com.samuelberrien.phyvr;

import android.content.Context;
import android.content.res.AssetManager;

public class MainWrappers {

	static {
		System.loadLibrary("phyvr");
	}

	private Context context;

	private long rendererPtr;
	private long enginePtr;
	private long levelPtr;

	private boolean vr;

	private boolean isFree;
	private boolean isInit;

	private int levelIdx;

	public MainWrappers(Context context, boolean vr, int levelIdx) {
		this.context = context;
		this.vr = vr;
		isFree = false;
		isInit = false;
		this.levelIdx = levelIdx;
	}

	public void init() {
		levelPtr = makeLevel(levelIdx);
		enginePtr = makeEngine(levelPtr, context.getAssets());
		initLevel(context.getAssets(), vr, levelPtr, enginePtr);
		rendererPtr = makeRenderer(levelPtr);
		isInit = true;
	}

	private void checkInit() {
		if (!isInit())
			throw new RuntimeException("C++ stuff is not initialized !");
	}

	public long getLevelPtr() {
		checkInit();
		return levelPtr;
	}

	public void update() {
		checkInit();
		updateEngine(enginePtr);
	}

	public boolean win() {
		checkInit();
		return hasWon(levelPtr);
	}

	public boolean lose() {
		checkInit();
		return hasLose(levelPtr);
	}

	public void willDraw(float[] mHeadView, boolean VR) {
		checkInit();
		willDrawRenderer(rendererPtr, mHeadView, VR);
	}

	public void draw(float[] mEyeProjectionMatrix,
					 float[] mEyeViewMatrix,
					 float[] myLighPosInEyeSpace,
					 float[] mCameraPos) {
		checkInit();
		drawRenderer(rendererPtr,
				mEyeProjectionMatrix, mEyeViewMatrix, myLighPosInEyeSpace, mCameraPos);
	}

	public boolean isFree() {
		return isFree;
	}

	public void free() {
		if (isInit) {
			freeEngine(enginePtr);
			freeRenderer(rendererPtr);
			freeLevel(levelPtr);
			isFree = true;
			isInit = false;
		} else {
			throw new RuntimeException("Wrappers are already free");
		}
	}

	public boolean isInit() { return isInit; }

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
