program reference_glm_hessian_driver
  use sklearn_glm_lbfgs,only:glm_raw_prediction
  use sklearn_glm_newton,only:glm_gradient_hessian
  use sklearn_common,only:dp
  use sklearn_sparse,only:csr_matrix,csr_from_dense
  use,intrinsic::ieee_arithmetic
  use,intrinsic::ieee_exceptions
  implicit none
  integer::input,output,count,case_id,n,p,q,i,j,k,intercept,repr
  real(dp)::power,alpha
  logical::warning
  character(16)::link
  character(1024)::path
  logical::before(size(ieee_all)),after(size(ieee_all)),hm(size(ieee_all)),ha(size(ieee_all))
  call get_command_argument(1,path);open(newunit=input,file=trim(path),status='old')
  call get_command_argument(2,path);open(newunit=output,file=trim(path),status='replace')
  read(input,*)count
  do case_id=1,count
    block
      real(dp),allocatable::x(:,:),y(:),weight(:),point(:),g(:),h(:,:),raw(:)
      type(csr_matrix)::csr
      read(input,*)n,p,power,link,alpha,intercept,repr
      q=p+intercept;allocate(x(n,p),y(n),weight(n),point(q),g(q),h(q,q),raw(n))
      read(input,*)point
      do i=1,n
        read(input,*)x(i,:),y(i),weight(i)
      end do
      if(repr==1)csr=csr_from_dense(x)
      if(repr==2)then
        csr%n_rows=n;csr%n_cols=p
        allocate(csr%indptr(n+1),csr%indices(2*n*p+n),csr%data(2*n*p+n));k=1;csr%indptr(1)=1
        do i=1,n
          do j=p,1,-1
            csr%indices(k:k+1)=j;csr%data(k:k+1)=[.25_dp*x(i,j),.75_dp*x(i,j)];k=k+2
          end do
          csr%indices(k)=1;csr%data(k)=0;k=k+1;csr%indptr(i+1)=k
        end do
      end if
      if(repr==0)then
        call glm_raw_prediction(point,raw,dense=x)
      else
        call glm_raw_prediction(point,raw,sparse=csr)
      end if
      do i=1,size(ieee_all)
        call ieee_get_flag(ieee_all(i),before(i));call ieee_get_halting_mode(ieee_all(i),hm(i))
      end do
      if(repr==0)then
        call glm_gradient_hessian(point,raw,y,weight,power,link,alpha,g,h,warning,dense=x)
      else
        call glm_gradient_hessian(point,raw,y,weight,power,link,alpha,g,h,warning,sparse=csr)
      end if
      do i=1,size(ieee_all)
        call ieee_get_flag(ieee_all(i),after(i));call ieee_get_halting_mode(ieee_all(i),ha(i))
      end do
      write(output,'(*(es25.16e3,1x))')g
      write(output,'(*(es25.16e3,1x))')reshape(h,[q*q])
      write(output,*)merge(1,0,warning),merge(1,0,all(before.eqv.after).and.all(hm.eqv.ha))
    end block
  end do
  close(input);close(output)
end program
