program test_lbfgsb_bounds
  use sklearn_common,only:dp,require
  use lbfgsb_module,only:setulb
  implicit none
  integer,parameter::n=1,m=5
  real(dp)::x(n),l(n),u(n),f,g(n),wa(2*m*n+5*n+11*m*m+8*m),dsave(29),target,answer
  integer::nbd(n),iwa(3*n),isave(44),kind,iterations
  logical::lsave(4)
  character(60)::task,csave
  do kind=0,3
    x=1;l=0;u=3;nbd=kind;target=2;answer=2
    if(kind==3)then
      u=1.5_dp;answer=1.5_dp
    end if
    wa=0;iwa=0;isave=0;dsave=0;lsave=.false.;task='START';csave='';f=0;g=0;iterations=0
    do
      call setulb(n,m,x,l,u,nbd,f,g,1.e7_dp,1.e-10_dp,wa,iwa,task,-1,csave,lsave,isave,dsave)
      if(task(:2)=='FG')then
        f=(x(1)-target)**2;g=2*(x-target)
      else if(task(:5)=='NEW_X')then
        iterations=iterations+1;call require(iterations<=50,'bounded L-BFGS iteration cap')
      else
        exit
      end if
    end do
    call require(task(:4)=='CONV'.and.abs(x(1)-answer)<1.e-9_dp,'L-BFGS free/lower/both/upper bounds')
  end do
  print *,'L-BFGS one-sided bound status evaluation: PASS'
end program
