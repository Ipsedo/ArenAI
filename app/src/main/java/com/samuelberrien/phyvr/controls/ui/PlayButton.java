package com.samuelberrien.phyvr.controls.ui;

import android.content.Context;
import android.content.res.TypedArray;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.support.v4.content.ContextCompat;
import android.util.AttributeSet;
import android.view.MotionEvent;
import android.view.View;

import com.samuelberrien.phyvr.R;

public class PlayButton extends View {

	private int kind;
	private float[] pts;
	private float[] scaledPts;
	private int color;
	private int colorAccent;

	private float strokeWidth;

	private Paint paint;

	private boolean touched;

	private int pointerId;

	private PlayButtonListener listener;

	public PlayButton(Context context, AttributeSet attrs) {
		super(context, attrs);
		init(context, attrs);
	}

	public PlayButton(Context context, AttributeSet attrs, int defStyleAttr) {
		super(context, attrs, defStyleAttr);
		init(context, attrs);
	}

	public PlayButton(Context context, AttributeSet attrs, int defStyleAttr, int defStyleRes) {
		super(context, attrs, defStyleAttr, defStyleRes);
		init(context, attrs);
	}

	private void init(Context context, AttributeSet attrs) {
		touched = false;
		pointerId = -1;

		TypedArray a = context.getTheme()
				.obtainStyledAttributes(attrs, R.styleable.PlayButton, 0, 0);

		kind = a.getInt(R.styleable.PlayButton_button_type, 0);

		color = a.getColor(R.styleable.PlayButton_color, ContextCompat.getColor(context, R.color.colorPrimary));
		colorAccent = a.getColor(R.styleable.PlayButton_color_accent, ContextCompat.getColor(context, R.color.colorAccent));

		strokeWidth = a.getDimension(R.styleable.PlayButton_button_stroke_width, 5.f);

		paint = new Paint();
		paint.setStyle(Paint.Style.STROKE);
		paint.setStrokeWidth(strokeWidth);

		switch (kind) {
			case 0: // Fire
				pts = new float[]{
						0.5f, 0.2f,
						0.6f, 0.4f,

						0.6f, 0.4f,
						0.5f + 0.125f, 0.42f,

						0.5f + 0.125f, 0.42f,
						0.5f + 0.125f, 0.75f,

						0.5f + 0.125f, 0.75f,
						0.5f - 0.125f, 0.75f,

						0.5f - 0.125f, 0.75f,
						0.5f - 0.125f, 0.42f,

						0.5f - 0.125f, 0.42f,
						0.4f, 0.4f,

						0.4f, 0.4f,
						0.5f, 0.2f
				};
				break;
			case 1: // Brake
				pts = new float[]{
						0.2f, 0.2f,
						0.8f, 0.2f,

						0.8f, 0.2f,
						0.8f, 0.8f,

						0.8f, 0.8f,
						0.2f, 0.8f,

						0.2f, 0.8f,
						0.2f, 0.2f
				};
				break;
			case 2: // Respawn
				pts = new float[]{
					0.1f, 0.5f,
						0.9f, 0.5f,
						0.5f, 0.1f,
						0.5f, 0.9f
				};
				break;
		}

		scaledPts = new float[pts.length];
	}

	public void setListener(PlayButtonListener listener) {
		this.listener = listener;
	}

	@Override
	protected void onDraw(Canvas canvas) {
		if (touched) paint.setColor(colorAccent);
		else paint.setColor(color);

		canvas.drawCircle(getWidth() / 2.f, getHeight() / 2.f, getWidth() / 2.f - strokeWidth / 2.f, paint);

		for (int i = 0; i < scaledPts.length; i+=2) {
			scaledPts[i] = pts[i] * getWidth();
			scaledPts[i + 1] = pts[i + 1] * getHeight();
		}
		canvas.drawLines(scaledPts, paint);
	}

	@Override
	public boolean performClick() {
		return super.performClick();
	}

	@Override
	public boolean onTouchEvent(MotionEvent event) {
		int currPointerId = event.getPointerId(event.getActionIndex());

		if (touched && currPointerId != pointerId) return false;

		switch (event.getAction()) {
			case MotionEvent.ACTION_DOWN:
				if (!touched) {
					touched = true;
					pointerId = currPointerId;

					if (listener != null) listener.clicked(true);

					performClick();
					invalidate();
				}
				break;
			case MotionEvent.ACTION_UP:
				if (touched && pointerId == currPointerId) {
					touched = false;
					pointerId = -1;

					if (listener != null) listener.clicked(false);

					invalidate();
				}
				break;
		}
		return true;
	}

	public interface PlayButtonListener {
		void clicked(boolean clicked);
	}
}
