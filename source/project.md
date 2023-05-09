---
description: 프로젝트
---

# /project

## DarkNet Source Code

* `Yolo`를 개발한 `Joseph Redmon`의 [DarkNet](https://github.com/pjreddie/darknet)을 분석합니다.
* DarkNet은 딥러닝을 C언어로 구현했기 때문에 딥러닝이 동작하는 과정을 자세히 코드 구현으로 살펴볼 수 있다고 생각 됩니다.
* 저는 C언어를 기본 문법만 알기 때문에 중간 중간 C언어를 이해하면서 넘어가겠습니다.
* GPU 부분은 마지막에 추가하겠습니다.

### Project

* 프로젝트 : 하나의 실행파일을 만들어 내기 위해 필요한 여러 개의 소스 파일과 헤더 파일 등을 하나로 묶어 놓은 것을 말합니다.
* `프로젝트이름.vcxproj` : 프로젝트, 실행파일이나 dll(동적 연결 라이브러리)등과 같은 파일을 만듭니다.
* 솔루션 : 여러 개의 프로젝트의 모임, 응용 프로그램마다 하나가 존재하고 프로젝트를 생성하는 과정에서 프로젝트와 동일한 이름으로 자동 생성됩니다.
* `솔루션이름.sln` : 솔루션, 프로젝트를 관리하는 파일 입니다.

## DarkNet 구성요소

```shell
/cfg
/data
/examples
/include
/python
/scripts
/src
LICENCE
Makefile
```

## LICENCE

```
                                  YOLO LICENSE
                             Version 2, July 29 2016

THIS SOFTWARE LICENSE IS PROVIDED "ALL CAPS" SO THAT YOU KNOW IT IS SUPER
SERIOUS AND YOU DON'T MESS AROUND WITH COPYRIGHT LAW BECAUSE YOU WILL GET IN
TROUBLE HERE ARE SOME OTHER BUZZWORDS COMMONLY IN THESE THINGS WARRANTIES
LIABILITY CONTRACT TORT LIABLE CLAIMS RESTRICTION MERCHANTABILITY. NOW HERE'S
THE REAL LICENSE:

0. Darknet is public domain.
1. Do whatever you want with it.
2. Stop emailing me about it!
```

YOLO 개발자다운 재미있는 `LICENCE` 입니다.

1. DarkNet은 공용 도메인 입니다.
2. 원하는 곳 어디에든 사용해도 좋습니다.
3. 이메일을 보내지 마세요
