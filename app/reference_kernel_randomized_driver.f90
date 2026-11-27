program reference_kernel_randomized_driver
  use sklearn_kernel_pca,only:kernel_pca
  use sklearn_common,only:dp
  implicit none
  type(kernel_pca)::model
  real(dp),allocatable::x(:,:),q(:,:),z(:,:),t(:,:),inverse(:,:)
  integer::n,p,m,kernel,solver,inverse_fit,remove,i,unit,out
  character(1024)::path
  character(16),parameter::names(6)=[character(16)::'linear','rbf','poly','cosine','precomputed','sigmoid']
  call get_command_argument(1,path);open(newunit=unit,file=trim(path),status='old')
  read(unit,*)n,p,m,model%n_components,kernel,solver,model%iterated_power,model%random_state, &
    inverse_fit,remove,model%gamma,model%coef0
  allocate(x(n,p),q(m,p))
  do i=1,n
    read(unit,*)x(i,:)
  end do
  do i=1,m
    read(unit,*)q(i,:)
  end do
  close(unit);model%kernel=names(kernel+1);model%degree=2
  model%fit_inverse_transform=inverse_fit==1;model%remove_zero_eig=remove==1
  if(solver==1)model%eigen_solver='randomized'
  z=model%fit_transform(x);t=model%transform(q)
  call get_command_argument(2,path);open(newunit=out,file=trim(path),status='replace')
  write(out,*)size(model%eigenvalues_)
  call vector(model%eigenvalues_);call vector(reshape(model%eigenvectors_,[size(model%eigenvectors_)]))
  call vector(reshape(z,[size(z)]));z=model%transform(x);call vector(reshape(z,[size(z)]))
  call vector(reshape(t,[size(t)]))
  if(model%fit_inverse_transform)then
    inverse=model%inverse_transform(t);call vector(reshape(inverse,[size(inverse)]))
  else
    write(out,'(a)')''
  end if
  close(out)
contains
  subroutine vector(a)
    real(dp),intent(in)::a(:)
    if(size(a)>0)then
      write(out,'(*(es26.17e3,1x))')a
    else
      write(out,'(a)')''
    end if
  end subroutine
end program
