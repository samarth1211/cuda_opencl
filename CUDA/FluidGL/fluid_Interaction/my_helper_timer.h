#pragma once
#include<Windows.h>
#include<vector>
#include<stdio.h>
#include<exception>

class IStopWatchTimer
{
public:
	IStopWatchTimer() {}
	virtual ~IStopWatchTimer() {}
public:
	virtual void start() = 0;
	virtual void stop() = 0;
	virtual void reset() = 0;

	virtual float getTime() = 0;
	virtual float getAverageTime() = 0;

};


class StopWatchWin : public IStopWatchTimer
{
private:

	LARGE_INTEGER start_time;
	LARGE_INTEGER end_time;
	float diff_time;
	float total_time;
	bool running;
	int clock_sessions;
	double freq;
	bool freq_set;

public:
	StopWatchWin() : start_time(), end_time(), diff_time(0.0f), total_time(0.0f), running(false), clock_sessions(0), freq(0), freq_set(false)
	{
		if (!freq_set)
		{
			LARGE_INTEGER temp;
			QueryPerformanceFrequency(reinterpret_cast<LARGE_INTEGER *>(&temp));

			freq = (static_cast<double>(temp.QuadPart)) / 1000.0f;

			freq_set = true;
		}
	}

	~StopWatchWin() {}

	void start()
	{
		QueryPerformanceCounter(reinterpret_cast<LARGE_INTEGER *>(&start_time));
		running = true;
	}

	void stop()
	{
		QueryPerformanceCounter(reinterpret_cast<LARGE_INTEGER *>(&end_time));
		diff_time = static_cast<float>(((static_cast<double>(end_time.QuadPart) -
			static_cast<double>(start_time.QuadPart)) /
			freq));

		total_time += diff_time;
		clock_sessions++;
		running = false;
	}

	void reset()
	{
		diff_time = 0.0f;
		total_time = 0.0f;
		clock_sessions = (int)0;

		if (running)
		{
			QueryPerformanceCounter( reinterpret_cast<LARGE_INTEGER *>(&start_time) );
		}

	}

	float getTime()
	{
		float ret_val = total_time;
		if (running)
		{
			LARGE_INTEGER temp;
			QueryPerformanceCounter(reinterpret_cast<LARGE_INTEGER*>(&temp) );

			ret_val += static_cast<float>(((static_cast<double>(temp.QuadPart) -
				static_cast<double>(start_time.QuadPart)) /
				freq));
		}

		return ret_val;
	}

	float getAverageTime()
	{
		return (clock_sessions>0)? (total_time/clock_sessions) :0.0f;
	}

};

/***********************************************************************/
inline bool sdkCreateTimer(IStopWatchTimer **timer_interface)
{
	*timer_interface = reinterpret_cast<IStopWatchTimer*>(new StopWatchWin());

	return (*timer_interface != NULL) ? true : false;
}

inline bool sdkDeleteTimer(IStopWatchTimer **timer_interface)
{
	if (*timer_interface)
	{
		delete *timer_interface;
		*timer_interface = NULL;
		return true;
	}
	return false;
}

inline bool sdkStartTimer(IStopWatchTimer **timer_interface)
{
	if (*timer_interface)
	{
		(*timer_interface)->start();
		return true;
	}
	return false;
}

inline bool sdkStopTimer(IStopWatchTimer **timer_interface)
{
	if (*timer_interface)
	{
		(*timer_interface)->stop();
		return true;
	}
	return false;
}

inline bool sdkResetTimer(IStopWatchTimer **timer_interface)
{
	if (*timer_interface)
	{
		(*timer_interface)->reset();
		return true;
	}
	return false;
}


inline float sdkGetAverageTimerValue(IStopWatchTimer **timer_interface)
{
	if (*timer_interface)
	{
		return (*timer_interface)->getAverageTime();
	}
	else
	{
		return 0.0f;
	}
}

inline float sdkGetTimerValue(IStopWatchTimer **timer_interface)
{
	if (*timer_interface) 
	{
		return (*timer_interface)->getTime();
	}
	else 
	{
		return 0.0f;
	}
}
