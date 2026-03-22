#include "mv_camera.h"

#include <cstring>

CMvCamera::CMvCamera() : m_hDevHandle(MV_NULL) {}

CMvCamera::~CMvCamera() {
    if (m_hDevHandle) {
        MV_CC_DestroyHandle(m_hDevHandle);
        m_hDevHandle = MV_NULL;
    }
}

int CMvCamera::GetSDKVersion() {
    return MV_CC_GetSDKVersion();
}

int CMvCamera::EnumDevices(unsigned int nTLayerType, MV_CC_DEVICE_INFO_LIST* pstDevList) {
    return MV_CC_EnumDevices(nTLayerType, pstDevList);
}

bool CMvCamera::IsDeviceAccessible(MV_CC_DEVICE_INFO* pstDevInfo, unsigned int nAccessMode) {
    return MV_CC_IsDeviceAccessible(pstDevInfo, nAccessMode);
}

int CMvCamera::Open(MV_CC_DEVICE_INFO* pstDeviceInfo) {
    if (pstDeviceInfo == MV_NULL) {
        return MV_E_PARAMETER;
    }
    if (m_hDevHandle) {
        return MV_E_CALLORDER;
    }

    int nRet = MV_CC_CreateHandle(&m_hDevHandle, pstDeviceInfo);
    if (nRet != MV_OK) {
        return nRet;
    }

    nRet = MV_CC_OpenDevice(m_hDevHandle);
    if (nRet != MV_OK) {
        MV_CC_DestroyHandle(m_hDevHandle);
        m_hDevHandle = MV_NULL;
        return nRet;
    }

    std::memcpy(&m_stDeviceInfo, pstDeviceInfo, sizeof(MV_CC_DEVICE_INFO));
    return nRet;
}

int CMvCamera::Close() {
    if (m_hDevHandle == MV_NULL) {
        return MV_E_HANDLE;
    }

    MV_CC_CloseDevice(m_hDevHandle);
    int nRet = MV_CC_DestroyHandle(m_hDevHandle);
    m_hDevHandle = MV_NULL;
    return nRet;
}

bool CMvCamera::IsDeviceConnected() {
    return MV_CC_IsDeviceConnected(m_hDevHandle);
}

int CMvCamera::RegisterImageCallBack(
    void(__stdcall* cbOutput)(unsigned char* pData, MV_FRAME_OUT_INFO_EX* pFrameInfo, void* pUser),
    void* pUser) {
    return MV_CC_RegisterImageCallBackEx(m_hDevHandle, cbOutput, pUser);
}

int CMvCamera::StartGrabbing() {
    return MV_CC_StartGrabbing(m_hDevHandle);
}

int CMvCamera::StopGrabbing() {
    return MV_CC_StopGrabbing(m_hDevHandle);
}

int CMvCamera::GetImageBuffer(MV_FRAME_OUT* pFrame, int nMsec) {
    return MV_CC_GetImageBuffer(m_hDevHandle, pFrame, nMsec);
}

int CMvCamera::FreeImageBuffer(MV_FRAME_OUT* pFrame) {
    return MV_CC_FreeImageBuffer(m_hDevHandle, pFrame);
}

int CMvCamera::GetOneFrameTimeout(
    unsigned char* pData,
    unsigned int nDataSize,
    MV_FRAME_OUT_INFO_EX* pFrameInfo,
    int nMsec) {
    return MV_CC_GetOneFrameTimeout(m_hDevHandle, pData, nDataSize, pFrameInfo, nMsec);
}

int CMvCamera::DisplayOneFrame(MV_DISPLAY_FRAME_INFO* pDisplayInfo) {
    return MV_CC_DisplayOneFrame(m_hDevHandle, pDisplayInfo);
}

int CMvCamera::SetImageNodeNum(unsigned int nNum) {
    return MV_CC_SetImageNodeNum(m_hDevHandle, nNum);
}

int CMvCamera::GetDeviceInfo(MV_CC_DEVICE_INFO* pstDevInfo) {
    if (!pstDevInfo) {
        return MV_E_PARAMETER;
    }

    std::memcpy(pstDevInfo, &m_stDeviceInfo, sizeof(MV_CC_DEVICE_INFO));
    return MV_OK;
}

int CMvCamera::GetGevAllMatchInfo(MV_MATCH_INFO_NET_DETECT* pMatchInfoNetDetect) {
    if (pMatchInfoNetDetect == MV_NULL) {
        return MV_E_PARAMETER;
    }

    MV_CC_DEVICE_INFO stDevInfo = {0};
    GetDeviceInfo(&stDevInfo);
    if (stDevInfo.nTLayerType != MV_GIGE_DEVICE) {
        return MV_E_SUPPORT;
    }

    MV_ALL_MATCH_INFO struMatchInfo = {0};
    struMatchInfo.nType = MV_MATCH_TYPE_NET_DETECT;
    struMatchInfo.pInfo = pMatchInfoNetDetect;
    struMatchInfo.nInfoSize = sizeof(MV_MATCH_INFO_NET_DETECT);
    std::memset(struMatchInfo.pInfo, 0, sizeof(MV_MATCH_INFO_NET_DETECT));
    return MV_CC_GetAllMatchInfo(m_hDevHandle, &struMatchInfo);
}

int CMvCamera::GetU3VAllMatchInfo(MV_MATCH_INFO_USB_DETECT* pMatchInfoUSBDetect) {
    if (pMatchInfoUSBDetect == MV_NULL) {
        return MV_E_PARAMETER;
    }

    MV_CC_DEVICE_INFO stDevInfo = {0};
    GetDeviceInfo(&stDevInfo);
    if (stDevInfo.nTLayerType != MV_USB_DEVICE) {
        return MV_E_SUPPORT;
    }

    MV_ALL_MATCH_INFO struMatchInfo = {0};
    struMatchInfo.nType = MV_MATCH_TYPE_USB_DETECT;
    struMatchInfo.pInfo = pMatchInfoUSBDetect;
    struMatchInfo.nInfoSize = sizeof(MV_MATCH_INFO_USB_DETECT);
    std::memset(struMatchInfo.pInfo, 0, sizeof(MV_MATCH_INFO_USB_DETECT));
    return MV_CC_GetAllMatchInfo(m_hDevHandle, &struMatchInfo);
}

int CMvCamera::GetIntValue(IN const char* strKey, OUT MVCC_INTVALUE_EX* pIntValue) {
    return MV_CC_GetIntValueEx(m_hDevHandle, strKey, pIntValue);
}

int CMvCamera::SetIntValue(IN const char* strKey, IN int64_t nValue) {
    return MV_CC_SetIntValueEx(m_hDevHandle, strKey, nValue);
}

int CMvCamera::GetEnumValue(IN const char* strKey, OUT MVCC_ENUMVALUE* pEnumValue) {
    return MV_CC_GetEnumValue(m_hDevHandle, strKey, pEnumValue);
}

int CMvCamera::SetEnumValue(IN const char* strKey, IN unsigned int nValue) {
    return MV_CC_SetEnumValue(m_hDevHandle, strKey, nValue);
}

int CMvCamera::GetEnumEntrySymbolic(IN const char* strKey, IN OUT MVCC_ENUMENTRY* pstEnumEntry) {
    return MV_CC_GetEnumEntrySymbolic(m_hDevHandle, strKey, pstEnumEntry);
}

int CMvCamera::SetEnumValueByString(IN const char* strKey, IN const char* sValue) {
    return MV_CC_SetEnumValueByString(m_hDevHandle, strKey, sValue);
}

int CMvCamera::GetFloatValue(IN const char* strKey, OUT MVCC_FLOATVALUE* pFloatValue) {
    return MV_CC_GetFloatValue(m_hDevHandle, strKey, pFloatValue);
}

int CMvCamera::SetFloatValue(IN const char* strKey, IN float fValue) {
    return MV_CC_SetFloatValue(m_hDevHandle, strKey, fValue);
}

int CMvCamera::GetBoolValue(IN const char* strKey, OUT bool* pbValue) {
    return MV_CC_GetBoolValue(m_hDevHandle, strKey, pbValue);
}

int CMvCamera::SetBoolValue(IN const char* strKey, IN bool bValue) {
    return MV_CC_SetBoolValue(m_hDevHandle, strKey, bValue);
}

int CMvCamera::GetStringValue(IN const char* strKey, MVCC_STRINGVALUE* pStringValue) {
    return MV_CC_GetStringValue(m_hDevHandle, strKey, pStringValue);
}

int CMvCamera::SetStringValue(IN const char* strKey, IN const char* strValue) {
    return MV_CC_SetStringValue(m_hDevHandle, strKey, strValue);
}

int CMvCamera::CommandExecute(IN const char* strKey) {
    return MV_CC_SetCommandValue(m_hDevHandle, strKey);
}

int CMvCamera::GetOptimalPacketSize(unsigned int* pOptimalPacketSize) {
    if (pOptimalPacketSize == MV_NULL) {
        return MV_E_PARAMETER;
    }

    int nRet = MV_CC_GetOptimalPacketSize(m_hDevHandle);
    if (nRet < MV_OK) {
        return nRet;
    }

    *pOptimalPacketSize = static_cast<unsigned int>(nRet);
    return MV_OK;
}

int CMvCamera::RegisterExceptionCallBack(
    void(__stdcall* cbException)(unsigned int nMsgType, void* pUser),
    void* pUser) {
    return MV_CC_RegisterExceptionCallBack(m_hDevHandle, cbException, pUser);
}

int CMvCamera::RegisterEventCallBack(
    const char* pEventName,
    void(__stdcall* cbEvent)(MV_EVENT_OUT_INFO* pEventInfo, void* pUser),
    void* pUser) {
    return MV_CC_RegisterEventCallBackEx(m_hDevHandle, pEventName, cbEvent, pUser);
}

int CMvCamera::ForceIp(unsigned int nIP, unsigned int nSubNetMask, unsigned int nDefaultGateWay) {
    return MV_GIGE_ForceIpEx(m_hDevHandle, nIP, nSubNetMask, nDefaultGateWay);
}

int CMvCamera::SetIpConfig(unsigned int nType) {
    return MV_GIGE_SetIpConfig(m_hDevHandle, nType);
}

int CMvCamera::SetNetTransMode(unsigned int nType) {
    return MV_GIGE_SetNetTransMode(m_hDevHandle, nType);
}

int CMvCamera::ConvertPixelType(MV_CC_PIXEL_CONVERT_PARAM* pstCvtParam) {
    return MV_CC_ConvertPixelType(m_hDevHandle, pstCvtParam);
}

int CMvCamera::ConvertPixelTypeEx(MV_CC_PIXEL_CONVERT_PARAM_EX* pstCvtParam) {
    return MV_CC_ConvertPixelTypeEx(m_hDevHandle, pstCvtParam);
}

int CMvCamera::SaveImage(MV_SAVE_IMAGE_PARAM_EX* pstParam) {
    return MV_CC_SaveImageEx2(m_hDevHandle, pstParam);
}
