program reference_lars_gram_driver
  use sklearn_least_angle,only:lars_path_gram,lars_path_result
  use sklearn_common,only:dp
  implicit none
  type(lars_path_result)::path
  real(dp),allocatable::gram(:,:),xy(:),saved(:,:),saved_xy(:)
  real(dp)::alpha,eps
  integer::input,output,n,p,limit,positive,i
  character(1024)::file
  character(8)::method
  call get_command_argument(1,file);open(newunit=input,file=trim(file),status='old')
  call get_command_argument(2,file);open(newunit=output,file=trim(file),status='replace')
  read(input,*)n,p,limit,alpha,eps,positive,method
  allocate(gram(p,p),xy(p))
  read(input,*)xy
  do i=1,p
    read(input,*)gram(i,:)
  end do
  saved=gram;saved_xy=xy
  path=lars_path_gram(xy,gram,n,method=trim(method),max_iter=limit,alpha_min=alpha,eps=eps,positive=positive==1)
  if(any(gram/=saved).or.any(xy/=saved_xy))error stop 'Gram inputs modified'
  write(output,'(a,*(1x,es25.16e3))')'alphas',path%alphas
  write(output,'(a,*(1x,es25.16e3))')'path',reshape(path%coefficients,[size(path%coefficients)])
  write(output,'(a,*(1x,i0))')'active',path%active
  write(output,'(a,1x,i0)')'iterations',path%n_iter
end program
